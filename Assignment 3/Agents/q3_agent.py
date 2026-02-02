# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
import random
import game
import util
import samples

from featureExtractors import EnhancedFeatureExtractor, FEATURE_NAMES
from q3Model import Q3Model

import numpy as np
from pacman import GameState

class Q3Agent(Agent):

    def __init__(self, model_path="./models/q3.model"):

        print('-------------Testing trained Model in Pacman-------------------')

        # We need the max and min feature values to scale new features to be within the same range
        self.max_values = np.loadtxt("./pacmandata/q3_max_feature_values.txt")
        self.min_values = np.loadtxt("./pacmandata/q3_min_feature_values.txt")

        self.model = Q3Model()
        self.model.load(model_path)

        # flag so we only precompute BFS distances once
        self.first_game = True

    def registerInitialState(self, gameState):

        # Use the same feature function to extract features from but only compute BFS distances once
        if self.first_game:
            self.first_game = False
            self.featureExtractor = EnhancedFeatureExtractor(startingGameState=gameState)


    def getAction(self, state: GameState):
        """
        Takes a game state object and selects an action for Pac-man using the trained model
        to determine the quality of each action.
        """
        features_and_actions = self.featureExtractor.extractFeatureForAllActions(state)[0]
        features_and_actions_numpy = {}
        for action, feature_dict in features_and_actions.items():
            feature_vector = np.array([feature_dict[feature_name] for feature_name in FEATURE_NAMES])
            feature_vector = (feature_vector - self.min_values) / (self.max_values - self.min_values)
            features_and_actions_numpy[action] = feature_vector
        return self.model.selectBestAction(features_and_actions_numpy)


