import random
import numpy as np
from collections import deque
import torch

from game import Environment, Move, BLOCK_SIZE, SUCCESSFEE, NO_PREDATORS
from model import DQN, QTrainer

NO_EPISODES = 1_000_000
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    # initialise all the parameters
    def __init__(self, id):
        self.n_games = 1
        self.epsilon = 0  # epsilon greedy exploration-exploitation
        self.gamma = 0.9  # discounting
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(8, 256, 4) # observation-hidden-decision
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.id = id

    # get state
    def get_state(self, game):
        point_pred = game.predators[self.id]

        point_r = game.movings(Move.RIGHT, point_pred)
        point_l = game.movings(Move.LEFT, point_pred)
        point_u = game.movings(Move.UP, point_pred)
        point_d = game.movings(Move.DOWN, point_pred)

        danger_r = int(game.is_out(point_r) or game.is_collision(point_r))
        danger_l = int(game.is_out(point_l) or game.is_collision(point_l))
        danger_u = int(game.is_out(point_u) or game.is_collision(point_u))
        danger_d = int(game.is_out(point_d) or game.is_collision(point_d))

        prey_r = int(game.predators[self.id].x < game.preys[0].x)
        prey_l = int(game.predators[self.id].x > game.preys[0].x)
        prey_u = int(game.predators[self.id].y > game.preys[0].y)
        prey_d = int(game.predators[self.id].y < game.preys[0].y)

        state = [danger_r,
                 danger_l,
                 danger_u,
                 danger_d,
                 prey_r,
                 prey_l,
                 prey_u,
                 prey_d]

        return np.array(state, dtype=int)

    # save one step
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # learn by one step
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # learn by a batch of steps
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # get action, explore or exploit
    def get_action(self, state):
        self.epsilon = 160 - self.n_games
        action = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


# main
if __name__ == '__main__':
    random.seed(1)
    cnt_episodes = 0

    # populate
    agents = []
    for i in range(NO_PREDATORS):
        agents.append(Agent(i))
    game = Environment()

    # the main loop
    while cnt_episodes < NO_EPISODES:
        # move prey
        game.move_prey(random.choice(list(Move)), 0)
        
        for pred_id in range(NO_PREDATORS):
            agent = agents[pred_id]

            # get old state
            state_old = agent.get_state(game)

            # get next action
            action = agent.get_action(state_old)

            # perform move, get new state
            reward, done = game.play_step(action, agent.id)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, action, reward, state_new, done)

            # save step
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                print('Game number:', agent.n_games)

                for pred_idq in range(NO_PREDATORS):
                    # inform the other agents on the outcome
                    if pred_id != pred_idq:
                        last = agents[pred_idq].memory[-1]
                        agents[pred_idq].memory.pop()
                        if reward > 0:
                            reward = reward / 3.0 # a smaller success fee
                        agents[pred_idq].remember(last[0], last[1], reward, last[3], done)
                    # train all
                    agents[pred_idq].train_long_memory()
                    agents[pred_idq].n_games += 1

        cnt_episodes = cnt_episodes + 1