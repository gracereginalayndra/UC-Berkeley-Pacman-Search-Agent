**README:**

# FIT3080 Pac-Man AI: From Search to Learning

This repository contains implementations for two FIT3080 (Artificial Intelligence) assignments at Monash University, Australia, inspired by the UC Berkeley CS188 (Introduction to AI) course assignment. The project focuses on developing intelligent Pac-Man agents using various AI techniques, demonstrating the progression from classical search algorithms to modern reinforcement learning and machine learning approaches.

## 📋 Project Overview

### **Assignment 1: Search Algorithms**

**Part 1: Single-Agent Search**
- **Q1(a):** A* search with Manhattan heuristic for single-dot navigation
- **Q1(b):** Custom search algorithms for multiple-dot collection

**Part 2: Adversarial Search**
- **Q2:** Alpha-beta pruning for Pac-Man with multiple ghosts


### **Assignment 3: Reinforcement Learning & Machine Learning**

**Part 1: Reinforcement Learning**
- **Q1:** Value/Policy Iteration for MDPs with stochastic actions
- **Q2:** Q-learning with epsilon-greedy exploration

**Part 2: Machine Learning**
- **Q3:** Supervised ML model for action prediction from game features

## 📁 Repository Structure

```
FIT3080-Pacman-AI-Search-to-Learning/
├── Assignment1/                    # Assignment 1: Search
│   ├── agents/
│   │   ├── q1a_solver.py           # A* solver (single dot)
│   │   ├── q1b_solver.py           # Multi-dot solver
│   │   └── q2_agent.py             # Alpha-beta adversarial agent
│   ├── layouts/                    # Maze layouts for Assignment 1
│   ├── reports/                    # Assignment 1 report
│   └── README_assignment1.md       # Assignment-specific instructions
│
├── Assignment3/                    # Assignment 3: RL & ML
│   ├── agents/
│   │   ├── q1_agent.py             # MDP value/policy iteration
│   │   ├── q2_agent.py             # Q-learning agent
│   │   └── q3_agent.py             # ML-based agent
│   ├── models/
│   │   ├── q3_model.py             # ML model definition
│   │   └── q3.model                # Saved trained model
│   ├── layouts/                    # Maze layouts for Assignment 3
│   ├── reports/                    # Assignment 3 report
│   └── README_assignment3.md       # Assignment-specific instructions
│
├── common/                         # Shared resources
│   ├── pacman.py                   # Game simulator (if shared)
│   └── utils.py                    # Common utilities
│
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## 📝 Key Features

### Search Algorithms (Assignment 1)
- **A* Search** with Manhattan heuristic
- **Custom search algorithms** for dot collection optimization
- **Alpha-Beta Pruning** for adversarial gameplay
- **State-space search** with performance optimization

### Learning Algorithms (Assignment 3)
- **Markov Decision Processes** with value/policy iteration
- **Q-learning** with epsilon-greedy exploration
- **State representation** design for efficient learning
- **Supervised ML models** for action prediction
- **Feature engineering** from game states

## ⚠️ Important Notes

1. **Submission:** Each assignment has separate submission instructions
2. **Code Modification:** Only modify files specified in assignment instructions
3. **Branches:** Assignment 3 uses a separate Git branch (`FIT3080_assignment3`)
4. **Reports:** Technical reports are submitted separately via Moodle
5. **AI Usage:** Include Generative AI Statement if AI tools were used

## 📚 Learning Progression

This repository demonstrates the evolution of AI techniques for game playing:

1. **Classical Search** → Heuristic search, optimal pathfinding
2. **Adversarial Search** → Multi-agent planning under competition
3. **Reinforcement Learning** → Learning from interaction, trial-and-error
4. **Supervised Learning** → Learning from demonstration data

## 🎯 Academic Context

**Monash University FIT3080 – Artificial Intelligence**  
*Unit Focus:* Problem solving as search, planning, learning, and reasoning under uncertainty
