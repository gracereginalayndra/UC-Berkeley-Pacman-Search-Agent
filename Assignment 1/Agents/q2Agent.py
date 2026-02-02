import logging
import random


import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def betterEvaluationFunction(currentGameState):
    """
    An optimized evaluation function that considers:
    - Current score (includes move penalties and rewards)
    - Distance to nearest food
    - Ghost states and distances (scared vs active)
    - Number of remaining food and capsules
    - Strategic positioning relative to power pellets
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
   
    # Extract current state information
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()
   
    # Food calculations
    foodList = foodGrid.asList()
    minFoodDist = float('inf')
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPosition, food) for food in foodList)
   
    # Ghost calculations with scared state consideration
    ghostScore = 0
    activeGhostsNearby = False
   
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pacmanPosition, ghostPos)
       
        if ghost.scaredTimer > 0:
            # Positive incentive for chasing scared ghosts
            if distance < ghost.scaredTimer:
                ghostScore += 250.0 / (distance + 1)  # Big reward for close scared ghosts
        else:
            # Active ghost - avoid but don't panic from a distance
            if distance < 2:
                ghostScore -= 500.0  # Strong penalty for very close ghosts
                activeGhostsNearby = True
            elif distance < 5:
                ghostScore -= 20.0 / distance  # Moderate penalty for nearby ghosts
   
    # Capsule calculations - encourage eating them when ghosts are threats
    capsuleScore = 0
    if activeGhostsNearby and capsules:
        minCapsuleDist = min(manhattanDistance(pacmanPosition, capsule) for capsule in capsules)
        capsuleScore = 150.0 / (minCapsuleDist + 1)  # Incentive to reach capsules when threatened
   
    # Food count penalty (encourage eating food)
    foodCountPenalty = -4 * len(foodList)
   
    # Capsule count penalty (encourage eating capsules)
    capsuleCountPenalty = -20 * len(capsules)
   
    # Combine all factors with appropriate weights
    evaluation = (
        currentScore +
        (-1.5 * minFoodDist) +
        ghostScore +
        capsuleScore +
        foodCountPenalty +
        capsuleCountPenalty
    )
   
    return evaluation


class Q2_Agent(Agent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.lastPositions = []  # Track recent positions to avoid cycling


    @log_function
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using alpha-beta pruning from the current gameState
        using self.depth and self.evaluationFunction.
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal state or maximum depth reached
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
           
            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
           
            # Pac-Man's turn (maximizing player)
            if agentIndex == 0:
                value = float('-inf')
               
                # Prioritize actions that don't lead to recent positions
                actionScores = []
                currentPos = state.getPacmanPosition()
               
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    newPos = successor.getPacmanPosition()
                   
                    # Penalize moving back to recent positions
                    positionPenalty = 0
                    if newPos in self.lastPositions:
                        positionPenalty = -50  # Discourage cycling through same positions
                   
                    # Evaluate the action with alpha-beta
                    actionValue = alphaBeta(successor, depth, 1, alpha, beta) + positionPenalty
                    actionScores.append((action, actionValue))
                   
                    # Update value and alpha
                    value = max(value, actionValue)
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
               
                return value
           
            # Ghosts' turn (minimizing players)
            else:
                value = float('inf')
               
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                   
                    # Check if this is the last ghost
                    if agentIndex == numAgents - 1:
                        # Next agent is Pac-Man, reduce depth
                        actionValue = alphaBeta(successor, depth - 1, 0, alpha, beta)
                    else:
                        # Next ghost
                        actionValue = alphaBeta(successor, depth, agentIndex + 1, alpha, beta)
                   
                    # Update value and beta
                    value = min(value, actionValue)
                    if value < alpha:
                        return value
                    beta = min(beta, value)
               
                return value


        # Main part of getAction
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
       
        # Get legal actions for Pac-Man
        legalActions = gameState.getLegalActions(0)
        currentPos = gameState.getPacmanPosition()
       
        # Update position history (keep only recent positions)
        self.lastPositions.append(currentPos)
        if len(self.lastPositions) > 5:
            self.lastPositions.pop(0)
       
        # Evaluate each action
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            newPos = successor.getPacmanPosition()
           
            # Penalize moving back to recent positions to prevent cycling
            positionPenalty = 0
            if newPos in self.lastPositions:
                positionPenalty = -50  # Significant penalty for cycling
               
            value = alphaBeta(successor, self.depth, 1, alpha, beta) + positionPenalty
           
            if value > bestValue:
                bestValue = value
                bestAction = action
               
            # Update alpha value
            alpha = max(alpha, bestValue)
       
        return bestAction if bestAction is not None else Directions.STOP