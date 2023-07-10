from collections import namedtuple
import pygame
import random
from enum import Enum
import numpy as np

pygame.init()


class Move(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (110, 110, 110)
RED = (200, 0, 0)
BLUE = (0, 0, 255)


BLOCK_SIZE = 20
SPEED = 500
NO_PREDATORS = 3
NO_PREY = 1
NO_OBSTACLES = 1
NO_STEPS = 10_000
SUCCESSFEE = 100_000
BOUNDARY_PENALTY = -500
STEPS_PENALTY = -5_000


def calculate_distance(pt1, pt2):
    return max(abs(pt1.x - pt2.x) / BLOCK_SIZE, abs(pt1.y - pt2.y) / BLOCK_SIZE)


def calculate_reward(pt, prey):
    return (SUCCESSFEE / (calculate_distance(pt, prey) + 1)) / SUCCESSFEE


class Environment:

    # initialise display, run reset()
    def __init__(self, w=320, h=240) -> None:
        self.obstacle = []
        self.scores = []
        for i in range(NO_PREDATORS):
            self.scores.append([])
        self.predators = None
        self.preys = None
        self.step = None
        self.score = None
        self.reward = 0
        self.ep = 0
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Predator and prey')
        self.clock = pygame.time.Clock()
        self._place_obstacle()
        self.reset()

    # position predators and prey, and obstacles
    def reset(self):
        self.predators = []
        self.preys = []
        for pred_id in range(NO_PREDATORS):
            x = self.w / 2 + (pred_id - 1) * BLOCK_SIZE
            y = self.h / 2 + (pred_id - 1) * BLOCK_SIZE
            self.predators.append(Point(x, y))
        for prey_id in range(NO_PREY):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.preys.append(Point(x, y))
        # extras
        self.step = 0
        self.reward = 0
        self.ep = self.ep + 1
        pygame.display.set_caption('Predator and prey ep ' + str(self.ep))
        pygame.display.flip()

    # the main game loop
    def play_step(self, action_tensor, pred_id):
        # 1. pygame events: quit
        self.step += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. recode a tensor to a Move
        if action_tensor == [1, 0, 0, 0]:
            action = Move.RIGHT
        elif action_tensor == [0, 1, 0, 0]:
            action = Move.LEFT
        elif action_tensor == [0, 0, 1, 0]:
            action = Move.UP
        else:
            action = Move.DOWN

        # 3. move the predator
        self._move(action, pred_id)

        # 4. check if the episode is over - no_steps reached or prey caught
        steps_end = self.step > NO_STEPS
        prey_caught = self.predators[pred_id] == self.preys[0]

        # return appropriate observation
        if prey_caught:
            self.reward = SUCCESSFEE
            game_over = True
            self.scores[pred_id].append(self.step)
            print(pred_id, ' has caught in: ', self.step, 'mean:' , sum(self.scores[pred_id]) / len(self.scores[pred_id]))
            return self.reward, game_over
        if steps_end:
            self.reward = STEPS_PENALTY
            game_over = True
            self.scores[pred_id].append(self.step)
            print(self.step, 'mean:' , sum(self.scores[pred_id]) / len(self.scores[pred_id]))
            return self.reward, game_over

        game_over = False

        # 5. update UI
        self._update_ui()
        self.clock.tick(SPEED)
        return self.reward, game_over

    # move the prey
    def move_prey(self, action, prey_id):
        old_pos = self.preys[prey_id]
        self.preys[prey_id] = self.movings(action, old_pos)
        if self.is_out(self.preys[prey_id]) or self.is_collision(self.preys[prey_id]):
            self.preys[prey_id] = old_pos

    # move the predator, set the reward
    def _move(self, action, pred_id):
        old_pos = self.predators[pred_id]
        self.predators[pred_id] = self.movings(action, old_pos)
        # do not fall out, penalise
        if self.is_out(self.predators[pred_id]) or self.is_collision(self.predators[pred_id]):
            self.predators[pred_id] = old_pos
            self.reward = BOUNDARY_PENALTY
        else:
            self.reward = calculate_reward(self.movings(action, old_pos), self.preys[0])

    # auxiliary function to return new position
    def movings(self, action, old_pos):
        new_pos = old_pos
        if action == Move.DOWN:
            new_pos = Point(old_pos.x, old_pos.y + BLOCK_SIZE)
        if action == Move.UP:
            new_pos = Point(old_pos.x, old_pos.y - BLOCK_SIZE)
        if action == Move.LEFT:
            new_pos = Point(old_pos.x - BLOCK_SIZE, old_pos.y)
        if action == Move.RIGHT:
            new_pos = Point(old_pos.x + BLOCK_SIZE, old_pos.y)
        return new_pos

    # tell if the point is outside the board
    def is_out(self, pt):
        return pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0
    
    # tell if the agent collides with the obstacle
    def is_collision(self, pt):
        if NO_OBSTACLES > 0:
            for obst in self.obstacle:
                if pt == obst:
                    return True
        return False
    
    # place one of the variants of the obstacle on the map
    def _place_obstacle(self):
        if NO_OBSTACLES > 0:
            pt = Point(BLOCK_SIZE, 0)
            length = 4
            for i in range(length):
                self.obstacle.append(pt)
                pt = self.movings(Move.DOWN, pt)

    # technical details of display
    def _update_ui(self):
        # the grid
        self.display.fill(GRAY)
        x = 0
        while x < self.w:
            y = 0
            while y < self.h:
                pygame.draw.rect(self.display, WHITE, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE), 1)
                y = y + BLOCK_SIZE
            x = x + BLOCK_SIZE

        # the objects
        for pred_id in range(NO_PREDATORS):
            pygame.draw.rect(self.display, (BLUE[0]+pred_id*100, BLUE[1]+pred_id*100, 255),
                             pygame.Rect(self.predators[pred_id].x, self.predators[pred_id].y, BLOCK_SIZE, BLOCK_SIZE))
        for prey_id in range(NO_PREY):
            pygame.draw.rect(self.display, RED,
                             pygame.Rect(self.preys[prey_id].x, self.preys[prey_id].y, BLOCK_SIZE, BLOCK_SIZE))
        for obst in self.obstacle:
            pygame.draw.rect(self.display, BLACK,
                             pygame.Rect(obst.x, obst.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.display.flip()