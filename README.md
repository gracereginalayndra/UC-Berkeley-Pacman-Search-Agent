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

## 🚀 Quick Start

### Assignment 1
```bash
# Single dot A* search
cd Assignment1
python pacman.py -l layouts/q1a_tinyMaze.lay -p SearchAgent -a fn=q1a_solver,prob=q1a_problem --timeout=1

# Multi-dot collection
python pacman.py -l layouts/q1b_tinySearch.lay -p SearchAgent -a fn=q1b_solver,prob=q1b_problem --timeout=10

# Adversarial search
python pacman.py -l layouts/q2_testClassic.lay -p Q2_Agent --timeout=30
```

### Assignment 3
```bash
# MDP value iteration
cd Assignment3
python pacman.py -l layouts/VI_smallMaze1.lay -p Q1Agent -a discount=0.9,iterations=100 -g StationaryGhost -n 40

# Q-learning
python pacman.py -l layouts/QL_small_1.lay -p Q2Agent -a epsilon=0.1,alpha=0.5,gamma=0.9 -x 1000 -n 1040

# Train ML model
python trainModel.py -m models/q3.model

# Test ML agent
python pacman.py -l layouts/ML_mediumClassic1.lay -p Q3Agent -a model_path=models/q3.model
```

## 📊 Evaluation & Grading

### Assignment 1
- **Q1(a):** 5 marks, evaluated on 5 instances with optimality ratio scoring
- **Q1(b):** 20 marks, evaluated on 20 instances relative to baseline
- **Q2:** 25 marks, evaluated on 25 adversarial instances
- **Report:** 30 marks (8-page technical report)

### Assignment 3
- **Q1:** 12 marks, MDP performance on small/medium/large mazes
- **Q2:** 21 marks, Q-learning performance with epsilon-greedy exploration
- **Q3:** 32 marks, ML model performance on 16 unseen mazes
- **Report:** 15 marks (6-page technical report)

## 🔧 Dependencies

**Assignment 1:**
- Python 3.x
- NumPy, SciPy (for Q1(a) only)

**Assignment 3:**
- Python 3.x
- NumPy
- *Note: Neural network libraries are NOT allowed for Q3*

Install all dependencies:
```bash
pip install numpy scipy
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
