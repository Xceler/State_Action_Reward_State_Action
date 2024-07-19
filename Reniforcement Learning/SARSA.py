import numpy as np 
import random 

class GridWorld:
    def __init__(self, grid_size, start, goal):
        self.grid_size = grid_size 
        self.start = start 
        self.goal = goal 
        self.state = start 
    
    def reset(self):
        self.state = start 
        return self.state 
    
    def step(self, action):
        x,y = self.state
        if action == 'up':
            x = max(0, x - 1)
        if action == 'down':
            x = min(self.grid_size - 1, x + 1)
        if action == 'left':
            y = max(0, y - 1)
        if action == 'right':
            y = min(self.grid_size - 1, y + 1)
        self.state = (x,y)
        reward  = - 1
        done = False 
        if self.state == self.goal:
            reward = 0
            done = True
        return self.state, reward, done 
    
    def get_possible_actions(self):
        return ['up', 'down', 'left', 'right']
    

class SARSAAgent:
    def __init__(self, env, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
        self.env = env 
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon 
        self.q_table = {}
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.env.get_possible_actions())
        else:
            q_values = [self.get_q_value(state,a) for a in self.env.get_possible_actions()]
            max_q = max(q_values)
            return random.choice([a for a , q in zip(self.env.get_possible_actions(), q_values) if q == max_q])
        
    def update_q_values(self, state, action ,reward, next_state, next_action):
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        new_q   = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[(state, action)] = new_q 
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            while True:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.update_q_values(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                if done:
                    break 
            
    def get_policy(self):
        policy = {}
        for state in [(x,y) for x in range(self.env.grid_size) for y in range(self.env.grid_size)]:
            action = self.choose_action(state)
            policy[state] = action 
        return policy 
    

grid_size = 5
start = (0 , 0)
goal = (4,4)
env = GridWorld(grid_size, start, goal)
agent = SARSAAgent(env)
agent.train(episodes = 500)
policy = agent.get_policy()
for state, action in policy.items():
    print(f"State: {state}, Action : {action}")
