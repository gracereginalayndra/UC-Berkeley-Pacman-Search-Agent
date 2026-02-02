#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

import heapq
def q1a_solver(problem: q1a_problem):
    # YOUR CODE HERE
    """
    A* search algorithm implementation
    """
    start = problem.getStartState()
    goal = problem.goal
    
    # Priority queue: (f_cost, state, path, cost_so_far)
    frontier = []
    heapq.heappush(frontier, (astar_heuristic(start, goal), start, [], 0))
    
    # Visited nodes to avoid cycles
    visited = set()
    
    while frontier:
        # Get node with lowest f-cost
        f_cost, current_state, path, g_cost = heapq.heappop(frontier)
        
        # Skip if already visited
        if current_state in visited:
            continue
            
        visited.add(current_state)
        
        # Check if goal reached
        if problem.isGoalState(current_state):
            return path  # Return the path to goal
        
        # Explore successors
        for next_state, action, step_cost in problem.getSuccessors(current_state):
            if next_state not in visited:
                new_g_cost = g_cost + step_cost
                new_h_cost = astar_heuristic(next_state, goal)
                new_f_cost = new_g_cost + new_h_cost
                new_path = path + [action]
                
                heapq.heappush(frontier, (new_f_cost, next_state, new_path, new_g_cost))
    
    return []  # No path found


def astar_heuristic(current, goal):
    # YOUR CODE HERE
    """
    Manhattan distance heuristic
    current: (x, y) tuple
    goal: (x, y) tuple
    """
    x1, y1 = current
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)
    
