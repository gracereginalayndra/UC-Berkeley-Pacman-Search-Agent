# PacmanValueIterationAgent.py
# -----------------------
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

#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#
import util

from agents.learningAgents import ValueEstimationAgent
from game import Grid, Actions, Directions
import math
from pacman import GameState
import random
import numpy as np
import json


class Q1Agent(ValueEstimationAgent):
    """
    Q1 agent takes a Markov decision process
    (see pacmanMDP.py) on initialization and 
    runs value iteration or policy iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp="PacmanMDP", discount=0.5, iterations=200, mazeSize=None):
        """
          Your agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will likely use:
              self.MDP.getStates()
              self.MDP.getPossibleActions(state)
              self.MDP.getTransitionStatesAndProbs(state, action)
              self.MDP.getReward(state, action)
              self.MDP.isTerminal(state)
        """
        mdp_func = util.import_by_name('./', mdp)
        self.mdp_func = mdp_func

        print('[Q1Agent] using mdp ' + mdp_func.__name__)
        
        # set discount factor and the number of training iterations
        if mazeSize:
            self.discount = self.getParams(mazeSize, "gamma")
            self.iterations = self.getParams(mazeSize, "iterations")
        else:
            self.discount = float(discount)
            self.iterations = int(iterations)            

        # initialise the values and policy. You will update them in solveMDP
        self.values = None
        self.policy = None

        # flag so we only solve the MDP once no matter how many games we play
        self.mdp_solved = False


    def getAction(self, gameState: GameState):
        """
        Returns the action to take at the a location according to the values

        Note: reaching any positive terminal state is considered winning the game and results in +500 points
        To achieve this we need the ReachedPositiveTerminalStateException because the game wouldn't noramlly end with food remaining
        """

        pacman_location = gameState.getPacmanPosition()
        if pacman_location in self.MDP.getFoodStates():
            raise util.ReachedPositiveTerminalStateException("Reached a Positive Terminal State")
        else:
            best_action = self.deriveActionFromLearntPolicy(pacman_location)
            return self.MDP.applyNoiseToAction(pacman_location, best_action)

    def registerInitialState(self, gameState: GameState):

        # set up the mdp with the agent starting state and solve it
        self.MDP = self.mdp_func(gameState)

        # only attempt to solve the MDP once
        if not self.mdp_solved:
            self.mdp_solved = True
            self.solveMDP()

    #-------------------#
    # DO NOT MODIFY END #
    #-------------------#

    def getParams(self, maze_size, param_name):
        """
        Add your maze parameters here 
        """
        params = {
            "small": {
                "gamma": 1,
                "iterations": 30
                },
            "medium": {
                "gamma": 1,
                "iterations": 100
            },
            "large": {
                "gamma": 1,
                "iterations": 100
            }
        }
        return params[maze_size][param_name]
    

    def solveMDP(self):
        """
        This function will solve the mdp instance the agent received on input.
        This is where you implement either policy iteration or value iteration
        You can access the 
        - mdp with self.MDP
        - discount factor with self.discount
        - num of iteratons with self.iterations
        """
        # Initialize values for all states to 0
        self.values = util.Counter()
        
        # Get all states from the MDP
        states = self.MDP.getStates()
        
        # Run value iteration for the specified number of iterations
        for iteration in range(self.iterations):
            # Create a new counter for updated values
            new_values = util.Counter()
            
            # Update value for each state
            for state in states:
                # Terminal states have value 0
                if self.MDP.isTerminal(state):
                    new_values[state] = 0
                    continue
                
                # Get possible actions from this state
                possible_actions = self.MDP.getPossibleActions(state)
                
                # If no actions available, value remains 0
                if not possible_actions:
                    new_values[state] = 0
                    continue
                
                # Compute Q-value for each action and take the max
                q_values = []
                for action in possible_actions:
                    q_value = self.computeQValueFromValues(state, action)
                    q_values.append(q_value)
                
                # Update state value to max Q-value
                new_values[state] = max(q_values)
            
            # Update values for next iteration
            self.values = new_values
        
        # After value iteration, compute the policy
        self.policy = {}
        for state in states:
            if self.MDP.isTerminal(state):
                self.policy[state] = None
            else:
                possible_actions = self.MDP.getPossibleActions(state)
                if possible_actions:
                    # Choose action with highest Q-value
                    best_action = None
                    best_value = float('-inf')
                    for action in possible_actions:
                        q_value = self.computeQValueFromValues(state, action)
                        if q_value > best_value:
                            best_value = q_value
                            best_action = action
                    self.policy[state] = best_action
                else:
                    self.policy[state] = None

    
    def deriveActionFromLearntPolicy(self, state: tuple):
        """
        This functions takes an (x,y) tuple representing Pac-Man's location
        and decides how to act using the agent's learnt policy.

        If you used Value Iteration this will be derived from the values you compute
        If you use Policy Iteration this will come directly from the computed policy
        """
        # Return the action from the computed policy
        return self.policy.get(state, None)


    ################################ ANY OTHER CODE BELOW HERE ################################
    
    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the value function stored in self.values.
        Q(s,a) = sum over s' of T(s,a,s') * [R(s,a,s') + gamma * V(s')]
        """
        q_value = 0.0
        
        # Get transition states and probabilities
        transitions = self.MDP.getTransitionStatesAndProbs(state, action)
        
        for next_state, prob in transitions:
            # Get reward for this transition
            reward = self.MDP.getReward(state, action, next_state)
            
            # Q-value += prob * (reward + discount * value of next state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        
        return q_value