# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import matplotlib.pyplot as plt
import math
from featureExtractors import FEATURE_NAMES
import pickle


class Q3Model:

    def __init__(self, learning_rate=0.075, epochs=2500, regularization=0.00008, batch_size=32, momentum=0.97):
        """
        Enhanced model with optimized hyperparameters for better gameplay
        """
        
        # Feature selection - carefully chosen for game performance
        feature_names_to_use = [
            'closestFoodNext', 
            'closestFoodNow',
            'closestCapsuleNext', 
            'closestCapsuleNow',
            'closestGhostNext',
            'closestGhostNow',
            'closestScaredGhostNext',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            'foodCount',
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',  
            'numberAvailableActions',
            'numberWallsSurroundingPacman',
            'ratioCapsuleDistance',
            'ratioFoodDistance',
            'ratioGhostDistance',
            'ratioScaredGhostDistance'
        ]
        
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(len(FEATURE_NAMES))))
        self.features_to_use = [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        # Model parameters - optimized for gameplay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.batch_size = batch_size
        self.momentum = momentum
        
        # Model weights
        self.weights = None
        self.bias = None
        
        # Momentum terms
        self.v_w = None
        self.v_b = None
        
        # Class weights
        self.pos_weight = None
        
        # Feature statistics
        self.feature_mean = None
        self.feature_std = None
        
        # Optimal threshold - tuned for game performance
        self.decision_threshold = 0.42

    def sigmoid(self, z):
        """Numerically stable sigmoid"""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def create_strategic_features(self, X):
        """Enhanced feature engineering for better gameplay decisions"""
        n_samples, n_features = X.shape if len(X.shape) > 1 else (1, X.shape[0])
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        features = [X]
        new_features = []
        
        if n_features >= 21:
            # Critical: Ghost danger assessment (indices 4, 5)
            ghost_danger = X[:, 4:5]  # closestGhostNext
            ghost_escape = np.maximum(0, X[:, 5:6] - X[:, 4:5])  # Positive if moving away
            new_features.append(ghost_escape * 2.0)  # Amplify escape reward
            
            # Immediate danger indicator
            immediate_danger = (ghost_danger < 0.15).astype(float) * 3.0
            new_features.append(immediate_danger)
            
            # Food pursuit efficiency (indices 0, 1)
            food_progress = np.maximum(0, X[:, 1:2] - X[:, 0:1])  # Getting closer
            new_features.append(food_progress * 2.5)
            
            # Critical action boosters
            eats_food_boost = X[:, 10:11] * 4.0  # Strong boost for eating food
            new_features.append(eats_food_boost)
            
            death_penalty = X[:, 8:9] * -8.0  # Extreme penalty for death
            new_features.append(death_penalty)
            
            capsule_boost = X[:, 9:10] * 3.5  # Strong boost for capsules
            new_features.append(capsule_boost)
            
            # Local food density - encourages efficient paths (indices 12, 13, 14)
            food_density = (X[:, 12:13] * 3 + X[:, 13:14] * 2 + X[:, 14:15]) / 3.0
            new_features.append(food_density)
            
            # Mobility and safety (indices 15, 16)
            mobility = X[:, 15:16] - X[:, 16:17] * 0.5
            new_features.append(mobility)
            
            # Scared ghost pursuit (indices 6, 7)
            scared_ghost_pursuit = np.maximum(0, X[:, 7:8] - X[:, 6:7])
            new_features.append(scared_ghost_pursuit * 1.5)
            
            # Food ratio boost (index 18)
            food_ratio_boost = (1.0 - X[:, 18:19]) * 1.2  # Favor getting closer
            new_features.append(food_ratio_boost)
            
            # Ghost ratio (index 19) - favor moving away
            ghost_ratio = X[:, 19:20]
            ghost_safety = np.where(ghost_ratio > 1.0, ghost_ratio * 1.5, ghost_ratio * 0.5)
            new_features.append(ghost_safety)
        
        if new_features:
            features.append(np.hstack(new_features))
        
        result = np.hstack(features)
        return result.reshape(-1) if n_samples == 1 else result

    def predict(self, feature_vector):
        """Predict quality of action from feature vector"""
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        if self.weights is None:
            raise Exception("Model not trained yet!")
        
        # Normalize
        if self.feature_mean is not None and self.feature_std is not None:
            vector_to_classify = (vector_to_classify - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Add strategic features
        vector_to_classify = self.create_strategic_features(vector_to_classify)
        
        # Compute prediction
        z = np.dot(vector_to_classify, self.weights) + self.bias
        probability = self.sigmoid(z)
        
        return probability

    def selectBestAction(self, features_and_actions):
        """
        Enhanced action selection with game-aware heuristics
        """
        best_action = None
        best_score = -float('inf')
        
        # Feature indices in original 23-feature vector
        idx = {
            'eatenByGhost': 8,
            'eatsCapsule': 9,
            'eatsFood': 10,
            'closestFoodNext': 0,
            'closestFoodNow': 1,
            'closestGhostNext': 4,
            'closestGhostNow': 5,
            'closestScaredGhostNext': 6,
            'closestScaredGhostNow': 7,
            'foodWithinThree': 12,
            'foodWithinFive': 13,
            'foodWithinNine': 14,
            'numActions': 15,
            'numWalls': 16,
            'ratioGhost': 19,
            'ratioScaredGhost': 20
        }
        
        action_scores = {}
        
        for action, feature_vector in features_and_actions.items():
            base_score = self.predict(feature_vector)
            adjusted_score = base_score
            
            # CRITICAL: Eliminate death actions
            if len(feature_vector) > idx['eatenByGhost']:
                if feature_vector[idx['eatenByGhost']] > 0.5:
                    adjusted_score *= 0.0001  # Essentially zero
                    action_scores[action] = adjusted_score
                    continue
            
            # HIGH PRIORITY: Food eating
            if len(feature_vector) > idx['eatsFood']:
                if feature_vector[idx['eatsFood']] > 0.5:
                    adjusted_score *= 1.5
            
            # HIGH PRIORITY: Capsule eating
            if len(feature_vector) > idx['eatsCapsule']:
                if feature_vector[idx['eatsCapsule']] > 0.5:
                    adjusted_score *= 1.6
            
            # Ghost avoidance - critical when close
            if len(feature_vector) > idx['closestGhostNow']:
                ghost_next = feature_vector[idx['closestGhostNext']]
                ghost_now = feature_vector[idx['closestGhostNow']]
                
                # Very close ghost - extreme caution
                if ghost_next < 0.1:
                    adjusted_score *= 0.3
                elif ghost_next < 0.2:
                    adjusted_score *= 0.6
                
                # Reward moving away from ghost
                if ghost_next > ghost_now:
                    boost = min((ghost_next - ghost_now) * 3.0, 1.3)
                    adjusted_score *= (1.0 + boost)
                # Penalize moving toward ghost when close
                elif ghost_next < ghost_now and ghost_now < 0.3:
                    adjusted_score *= 0.5
            
            # Food pursuit - reward getting closer
            if len(feature_vector) > idx['closestFoodNow']:
                food_next = feature_vector[idx['closestFoodNext']]
                food_now = feature_vector[idx['closestFoodNow']]
                
                if food_next < food_now:
                    progress = (food_now - food_next)
                    adjusted_score *= (1.0 + progress * 2.0)
            
            # Scared ghost pursuit - opportunity for points
            if len(feature_vector) > idx['closestScaredGhostNow']:
                scared_next = feature_vector[idx['closestScaredGhostNext']]
                scared_now = feature_vector[idx['closestScaredGhostNow']]
                
                # If scared ghost is close, pursue it
                if scared_next < 0.3 and scared_next < scared_now:
                    adjusted_score *= 1.4
            
            # Local food density - prefer food-rich areas
            if len(feature_vector) > idx['foodWithinNine']:
                food_nearby = (feature_vector[idx['foodWithinThree']] + 
                              feature_vector[idx['foodWithinFive']] + 
                              feature_vector[idx['foodWithinNine']])
                if food_nearby > 2:
                    adjusted_score *= 1.12
            
            # Mobility - avoid getting trapped
            if len(feature_vector) > idx['numActions']:
                available = feature_vector[idx['numActions']]
                walls = feature_vector[idx['numWalls']]
                
                if available < 0.4:  # Low mobility
                    adjusted_score *= 0.85
                if walls > 0.6:  # High walls
                    adjusted_score *= 0.9
            
            action_scores[action] = adjusted_score
        
        # Strong penalty for STOP action
        if Directions.STOP in action_scores:
            action_scores[Directions.STOP] *= 0.15
        
        # Select best action
        best_action = max(action_scores, key=action_scores.get)
        
        return best_action

    def evaluate(self, data, labels):
        """Evaluate model performance"""
        X_eval = data[:, self.features_to_use]

        if self.feature_mean is not None and self.feature_std is not None:
            X_eval = (X_eval - self.feature_mean) / (self.feature_std + 1e-8)
        
        X_eval = self.create_strategic_features(X_eval)
        
        predictions = []
        probabilities = []
        for i in range(len(X_eval)):
            z = np.dot(X_eval[i], self.weights) + self.bias
            prob = self.sigmoid(z)
            probabilities.append(prob)
            pred = 1 if prob >= self.decision_threshold else 0
            predictions.append(pred)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        accuracy = np.mean(predictions == labels)
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Decision Threshold: {self.decision_threshold:.4f}")
        
        return accuracy

    def find_optimal_threshold(self, X, y):
        """Find optimal decision threshold"""
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        
        best_score = 0
        best_threshold = 0.42
        
        for threshold in np.arange(0.35, 0.55, 0.01):
            preds = (probabilities >= threshold).astype(int)
            
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            tn = np.sum((preds == 0) & (y == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            
            # F2 score - strongly favor recall (finding good actions)
            beta = 2.0
            f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0
            
            # Weighted score emphasizing recall and F-beta
            combined_score = 0.7 * f_beta + 0.3 * accuracy
            
            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
        
        return best_threshold

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """Train the model with enhanced techniques"""
        X_train = trainingData[:, self.features_to_use]
        y_train = trainingLabels
        
        if validationData is not None:
            X_validate = validationData[:, self.features_to_use]
            y_validate = validationLabels

        # Compute normalization statistics
        self.feature_mean = np.mean(X_train, axis=0)
        self.feature_std = np.std(X_train, axis=0)
        
        # Normalize
        X_train = (X_train - self.feature_mean) / (self.feature_std + 1e-8)
        X_train = self.create_strategic_features(X_train)
        
        if validationData is not None:
            X_validate = (X_validate - self.feature_mean) / (self.feature_std + 1e-8)
            X_validate = self.create_strategic_features(X_validate)
        
        # Initialize weights
        n_features = X_train.shape[1]
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        
        self.v_w = np.zeros(n_features)
        self.v_b = 0.0
        
        n_samples = X_train.shape[0]
        
        # Enhanced class weighting for better recall
        n_pos = np.sum(y_train)
        n_neg = n_samples - n_pos
        self.pos_weight = (n_neg / n_pos) * 1.6 if n_pos > 0 else 1.0
        
        print(f"Class distribution - Positive: {n_pos} ({n_pos/n_samples:.2%}), Negative: {n_neg} ({n_neg/n_samples:.2%})")
        print(f"Positive class weight: {self.pos_weight:.2f}")
        print(f"Number of features: {n_features}")
        
        # Training with enhanced settings
        best_val_score = 0
        best_weights = None
        best_bias = None
        patience = 70
        patience_counter = 0
        min_delta = 0.0003
        
        for epoch in range(self.epochs):
            # Cosine annealing
            cycle = 175
            t = epoch % cycle
            current_lr = self.learning_rate * (0.5 + 0.5 * np.cos(np.pi * t / cycle))
            current_lr = max(current_lr, self.learning_rate * 0.03)
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                z = np.dot(X_batch, self.weights) + self.bias
                predictions = self.sigmoid(z)
                
                sample_weights = np.where(y_batch == 1, self.pos_weight, 1.0)
                weighted_error = (predictions - y_batch) * sample_weights
                
                batch_size_actual = X_batch.shape[0]
                dw = (1/batch_size_actual) * np.dot(X_batch.T, weighted_error) + self.regularization * self.weights
                db = (1/batch_size_actual) * np.sum(weighted_error)
                
                dw = np.clip(dw, -5.0, 5.0)
                db = np.clip(db, -5.0, 5.0)
                
                self.v_w = self.momentum * self.v_w - current_lr * dw
                self.v_b = self.momentum * self.v_b - current_lr * db
                
                self.weights += self.v_w
                self.bias += self.v_b
            
            # Progress reporting
            if (epoch + 1) % 35 == 0 or epoch == 0:
                z_all = np.dot(X_train, self.weights) + self.bias
                pred_probs = self.sigmoid(z_all)
                
                temp_threshold = self.find_optimal_threshold(X_train, y_train)
                train_preds = (pred_probs >= temp_threshold).astype(int)
                
                train_acc = np.mean(train_preds == y_train)
                
                tp = np.sum((train_preds == 1) & (y_train == 1))
                fp = np.sum((train_preds == 1) & (y_train == 0))
                fn = np.sum((train_preds == 0) & (y_train == 1))
                
                train_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                train_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0
                
                loss = -np.mean(y_train * np.log(pred_probs + 1e-15) + 
                               (1 - y_train) * np.log(1 - pred_probs + 1e-15))
                
                print(f"Epoch {epoch+1}/{self.epochs}, LR: {current_lr:.5f}, Loss: {loss:.4f}, Acc: {train_acc:.4f}, P: {train_prec:.4f}, R: {train_rec:.4f}, F1: {train_f1:.4f}", end="")
                
                if validationData is not None:
                    val_z = np.dot(X_validate, self.weights) + self.bias
                    val_probs = self.sigmoid(val_z)
                    val_preds = (val_probs >= temp_threshold).astype(int)
                    val_acc = np.mean(val_preds == y_validate)
                    
                    tp_val = np.sum((val_preds == 1) & (y_validate == 1))
                    fp_val = np.sum((val_preds == 1) & (y_validate == 0))
                    fn_val = np.sum((val_preds == 0) & (y_validate == 1))
                    
                    val_prec = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
                    val_rec = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
                    val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
                    
                    val_combined_score = 0.65 * val_f1 + 0.35 * val_acc
                    
                    print(f" | Val Acc: {val_acc:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}, Score: {val_combined_score:.4f}")
                    
                    if val_combined_score > best_val_score + min_delta:
                        best_val_score = val_combined_score
                        best_weights = self.weights.copy()
                        best_bias = self.bias
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
                else:
                    print()
        
        # Find optimal threshold
        if validationData is not None:
            self.decision_threshold = self.find_optimal_threshold(X_validate, y_validate)
        else:
            self.decision_threshold = self.find_optimal_threshold(X_train, y_train)
        
        print(f"Training complete! Optimal threshold: {self.decision_threshold:.4f}")

    def save(self, weights_path):
        """Save model to file"""
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'features_to_use': self.features_to_use,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'regularization': self.regularization,
            'batch_size': self.batch_size,
            'momentum': self.momentum,
            'pos_weight': self.pos_weight,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'v_w': self.v_w,
            'v_b': self.v_b,
            'decision_threshold': self.decision_threshold
        }
        
        with open(weights_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {weights_path}")

    def load(self, weights_path):
        """Load model from file"""
        with open(weights_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.features_to_use = model_data['features_to_use']
        self.learning_rate = model_data.get('learning_rate', self.learning_rate)
        self.epochs = model_data.get('epochs', self.epochs)
        self.regularization = model_data.get('regularization', self.regularization)
        self.batch_size = model_data.get('batch_size', self.batch_size)
        self.momentum = model_data.get('momentum', self.momentum)
        self.pos_weight = model_data.get('pos_weight', None)
        self.feature_mean = model_data.get('feature_mean', None)
        self.feature_std = model_data.get('feature_std', None)
        self.v_w = model_data.get('v_w', None)
        self.v_b = model_data.get('v_b', None)
        self.decision_threshold = model_data.get('decision_threshold', 0.42)
        
        print(f"Model loaded from {weights_path}")