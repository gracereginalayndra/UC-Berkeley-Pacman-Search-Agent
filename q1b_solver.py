#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

import time
from game import Directions, Actions

def find_path_to_closest_food(problem, start_pos, food_set):
    from util import Queue
    visited = set()
    queue = Queue()
    queue.push((start_pos, []))  # store position and path actions
    visited.add(start_pos)
    
    while not queue.isEmpty():
        pos, path = queue.pop()
        if pos in food_set:
            return pos, path
            
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x = int(pos[0] + dx)
            next_y = int(pos[1] + dy)
            next_pos = (next_x, next_y)
            if not problem.walls[next_x][next_y] and next_pos not in visited:
                visited.add(next_pos)
                new_path = path + [action]
                queue.push((next_pos, new_path))
                
    return None, []

def q1b_solver(problem: q1b_problem):
    start_state = problem.getStartState()
    current_pos, food_set = start_state
    plan = []
    current_food_set = set(food_set)  # mutable set
    
    while current_food_set:
        food_pos, path = find_path_to_closest_food(problem, current_pos, current_food_set)
        if food_pos is None:
            break
        plan.extend(path)
        current_pos = food_pos
        current_food_set.remove(food_pos)
        
    return plan