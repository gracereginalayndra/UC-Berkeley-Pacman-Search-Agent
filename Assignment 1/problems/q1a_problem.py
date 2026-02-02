import logging
import time
from typing import Tuple

import util
from game import Actions, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1a_problem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        gameState: A GameState object (pacman.py)

        Store anything else that you think would helpful for your agent
        """
        self.startingGameState: GameState = gameState
        "*** YOUR CODE HERE ***"
        self.walls = gameState.getWalls()  # Grid of walls
        self.start = gameState.getPacmanPosition()  # Starting position
        self.food = gameState.getFood()  # Food grid
        self.food_positions = []  # List of food positions

        # Extract food positions from food grid
        width = self.food.width
        height = self.food.height
        for x in range(width):
            for y in range(height):
                if self.food[x][y]:
                    self.food_positions.append((x, y))

        # For single dot, we'll take the first food position
        self.goal = self.food_positions[0] if self.food_positions else None

    @log_function
    def getStartState(self):
        "*** YOUR CODE HERE ***"
        return self.start  # Return starting position
        util.raiseNotDefined()


    @log_function
    def isGoalState(self, state):
        "*** YOUR CODE HERE ***"
        return state == self.goal  # Check if current position is goal
        util.raiseNotDefined()

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        # ------------------------------------------
        "*** YOUR CODE HERE ***"
        successors = []
        x, y = state
        
        # Check all possible directions
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            next_state = (next_x, next_y)
            
            # Check if this move is valid (not a wall)
            if not self.walls[next_x][next_y]:
                successors.append((next_state, action, 1))  # Cost is always 1
        
        return successors
        util.raiseNotDefined()


