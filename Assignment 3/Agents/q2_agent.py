# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from agents.learningAgents import ReinforcementAgent
from pacman import GameState

import random,util,math
import numpy as np
from game import Directions
import json


class Q2Agent(ReinforcementAgent):
    """
      Q-Learning Agent

      Methods you should fill in:
        - __init__
        - registerInitialState
        - update
        - getParams
        - epsilonGreedyActionSelection

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """

    def __init__(self, usePresetParams=False, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p Q2Agent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes

        usePresetParams - Set to True if you want to use your evaluation parameters
        """

        self.index = 0  # This is always Pacman

        ReinforcementAgent.__init__(self, **args)

        # when maze size is provided use the preset parameters, otherwise values will use those specified at the command line
        if usePresetParams:
            self.epsilon = self.getParams("epsilon")
            self.alpha = self.getParams("alpha")
            self.discount = self.getParams("gamma")

        # *** YOUR CODE STARTS HERE ***

        # Initialize Q-table as a dictionary with default value 0
        # Key: (state, action), Value: Q-value
        self.Qtable = util.Counter()

        # *** YOUR CODE ENDS HERE ***
    

    def registerInitialState(self, state: GameState):
        """
        Don't modify this method except in the provided area.
        You can modify this method to do any computation you need at the start of each episode
        """

        # *** YOUR CODE STARTS HERE ***

        # No special initialization needed at the start of each episode
        # Q-table persists across episodes for learning
        pass

        # *** YOUR CODE ENDS HERE ***

        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))
    
    
    def getAction(self, state: GameState):
        """
        Don't modify this method!
        Uses epsilon greedy to select an action based on the agents Q table.
        """

        action = self.epsilonGreedyActionSelection(state)
        self.doAction(state, action)
        return action

    def update(self, state: GameState, action: str, nextState: GameState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here using the Q value update equation

        NOTE: You should never call this function,
        it will be called on your behalf
        """

        # *** YOUR CODE HERE ***
        
        # Convert state to a hashable representation
        currentState = self.getStateRepresentation(state)
        nextStateRep = self.getStateRepresentation(nextState)
        
        # Get current Q-value
        currentQValue = self.Qtable[(currentState, action)]
        
        # Calculate max Q-value for next state
        legalActions = self.getLegalActions(nextState)
        
        # Remove STOP from legal actions when calculating max Q
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        if len(legalActions) == 0:
            # Terminal state
            maxNextQValue = 0.0
        else:
            maxNextQValue = max([self.Qtable[(nextStateRep, nextAction)] 
                                for nextAction in legalActions])
        
        # Standard Q-learning update rule
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        self.Qtable[(currentState, action)] = currentQValue + self.alpha * (
            reward + self.discount * maxNextQValue - currentQValue
        )

    def getParams(self, param_name):
        """
        Add your parameters here 
        """
        params = {
            "gamma": 0.95,
            "epsilon": 0.15,
            "alpha": 0.225
        }
        return params[param_name]


    def epsilonGreedyActionSelection(self, state: GameState):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: When the agent is no longer in training self.epsilon will be set to 0, 
        so calling this method should always return the best action over the Q values
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        HINT: You might want to use self.getLegalActions(state), 
        but consider whether or not using the STOP action is necessary or beneficial
        """
        
        # *** YOUR CODE HERE ***
        
        # Get legal actions, removing STOP as it's generally not beneficial
        legalActions = self.getLegalActions(state)
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        # Terminal state check
        if len(legalActions) == 0:
            return None
        
        # Epsilon-greedy selection
        if util.flipCoin(self.epsilon):
            # Explore: choose random action
            return random.choice(legalActions)
        else:
            # Exploit: choose best action based on Q-values
            stateRep = self.getStateRepresentation(state)
            
            # Get Q-values for all legal actions
            qValues = [(self.Qtable[(stateRep, action)], action) 
                      for action in legalActions]
            
            # Get maximum Q-value
            maxQValue = max(qValues, key=lambda x: x[0])[0]
            
            # Get all actions with maximum Q-value (for tie-breaking)
            bestActions = [action for qval, action in qValues if qval == maxQValue]
            
            # If all Q-values are 0 (unexplored), use smart heuristics
            if maxQValue == 0.0:
                pacmanPos = state.getPacmanPosition()
                foodList = state.getFood().asList()
                ghostStates = state.getGhostStates()
                
                actionScores = []
                for action in legalActions:
                    successor = state.generateSuccessor(0, action)
                    newPos = successor.getPacmanPosition()
                    
                    score = 0
                    
                    # Food attraction
                    if len(foodList) > 0:
                        minFoodDist = min([abs(newPos[0] - food[0]) + abs(newPos[1] - food[1]) 
                                          for food in foodList])
                        score -= minFoodDist * 3  # Strong attraction to food
                    
                    # Ghost repulsion
                    if len(ghostStates) > 0:
                        ghostPos = ghostStates[0].getPosition()
                        ghostDist = abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
                        
                        if ghostDist <= 1:
                            score -= 10000  # Extremely dangerous
                        elif ghostDist == 2:
                            score -= 100  # Very dangerous
                        elif ghostDist == 3:
                            score -= 20  # Somewhat dangerous
                        else:
                            score += ghostDist  # Slight preference for being farther
                    
                    actionScores.append((score, action))
                
                # Return best action based on heuristic
                bestScore = max(actionScores, key=lambda x: x[0])[0]
                bestActions = [action for score, action in actionScores if score == bestScore]
            
            # Randomly choose among best actions
            return random.choice(bestActions)

    ################################ ANY OTHER CODE BELOW HERE ################################
    
    def getStateRepresentation(self, state: GameState):
        """
        Simplified but effective state representation
        """
        
        # Get Pacman position
        pacmanPos = state.getPacmanPosition()
        
        # Get food positions as sorted tuple
        foodList = state.getFood().asList()
        foodTuple = tuple(sorted(foodList))
        
        # Simplified ghost handling - only track when relevant
        ghostStates = state.getGhostStates()
        if len(ghostStates) > 0:
            ghostPos = ghostStates[0].getPosition()
            ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
            
            # Calculate Manhattan distance
            ghostDist = abs(pacmanPos[0] - ghostPos[0]) + abs(pacmanPos[1] - ghostPos[1])
            
            # Three-tier system
            if ghostDist <= 3:
                # Danger zone - track exact relative position
                relativeGhost = (ghostPos[0] - pacmanPos[0], ghostPos[1] - pacmanPos[1])
            elif ghostDist <= 6:
                # Medium zone - coarse relative position
                dx = ghostPos[0] - pacmanPos[0]
                dy = ghostPos[1] - pacmanPos[1]
                # Bin into larger zones
                dx_bin = 3 if dx > 2 else (-3 if dx < -2 else dx)
                dy_bin = 3 if dy > 2 else (-3 if dy < -2 else dy)
                relativeGhost = (dx_bin, dy_bin)
            else:
                # Safe zone
                relativeGhost = (100, 100)
        else:
            relativeGhost = (100, 100)
        
        # Create state tuple
        stateRep = (pacmanPos, foodTuple, relativeGhost)
        
        return stateRep