import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState

class q1b_problem:
    def __init__(self, gameState: GameState):
        self.startingGameState = gameState
        self.walls = gameState.getWalls()
        self.food = gameState.getFood()
        
        # Convert food to a set of coordinates
        self.food_positions = set()
        for x in range(self.food.width):
            for y in range(self.food.height):
                if self.food[x][y]:
                    self.food_positions.add((x, y))
                    
    @log_function
    def getStartState(self):
        start_pos = self.startingGameState.getPacmanPosition()
        return (start_pos, frozenset(self.food_positions))

    @log_function
    def isGoalState(self, state):
        return len(state[1]) == 0

    @log_function
    def getSuccessors(self, state):
        current_pos, food_set = state
        successors = []
        
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x = int(current_pos[0] + dx)
            next_y = int(current_pos[1] + dy)
            
            if not self.walls[next_x][next_y]:
                next_pos = (next_x, next_y)
                new_food_set = set(food_set)
                
                if next_pos in new_food_set:
                    new_food_set.remove(next_pos)
                
                successors.append(((next_pos, frozenset(new_food_set)), action, 1))
                
        return successors