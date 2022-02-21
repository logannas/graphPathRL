import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pylab as pl

import random

random.seed(100)
np.random.seed(50)

class graphRLnx(gym.Env):

    def __init__(self):
        self.start_state = 0
        self.aim_state = [4]
        self.fire = [3]
        
        self.current_state = self.start_state
        self.reward = 0
                
        self.time_step = 0
        self.done = False
        
        edges = [(0,1), (0,3), (0,2), (1,0), (1,2), (1,3), (2,0), (1,2), (2,4), (3,0), (3,1), (3,4)]
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
                
        #Drawn graph
        color_map = []
        for node in G:
            if node in self.fire:
                color_map.append('red')
            elif node in self.aim_state:
                color_map.append('green')
            else:
                color_map.append('blue')
                            
        nx.draw_networkx_nodes(G, pos, node_color=color_map)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
                
        pl.show()
        
        self.graph = G
        self.adjacent_mat = nx.adjacency_matrix(self.graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(self.graph, nodelist=range(self.num_nodes)).toarray()#:D

        
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))  # num_states * num_actions
        
        self.reset(self.start_state, self.aim_state, self.fire)

    def step(self, action):
        epsilon=0.05 
        alpha=0.1
        gamma=0.8
        
        next_state = self.epsilon_greedy(self.current_state, self.q_table, epsilon=epsilon)
        s_next_next = self.epsilon_greedy(next_state, self.q_table, epsilon=-0.2)  # epsilon<0, greedy policy
        if next_state in self.fire:
                    reward = -10
        else:
            reward = -self.adjacent_mat[self.current_state][next_state]
        delta = reward + gamma * self.q_table[next_state, s_next_next] - self.q_table[self.current_state, next_state]
        
        self.q_table[self.current_state, next_state] = self.q_table[self.current_state, next_state] + alpha * delta
        # update current state
        self.current_state = next_state 

        if self.current_state in self.aim_state:
            self.done = True
        return self.current_state, reward, self.done, {"time_step": self.time_step}

    def reset(self, start_state, aim_state, fire):
        self.start_state = start_state
        self.aim_state = aim_state
        self.fire = fire
        
        self.current_state = self.start_state
        self.reward = 0
                
        self.time_step = 0
        self.done = False
        
        edges = [(0,1), (0,3), (0,2), (1,0), (1,2), (1,3), (2,0), (1,2), (2,4), (3,0), (3,1), (3,4)]
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
                
        #Drawn graph
        color_map = []
        for node in G:
            if node in self.fire:
                color_map.append('red')
            elif node in self.aim_state:
                color_map.append('green')
            else:
                color_map.append('blue')
                            
        nx.draw_networkx_nodes(G, pos, node_color=color_map)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
                
        self.graph = G
        self.adjacent_mat = nx.adjacency_matrix(self.graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(self.graph, nodelist=range(self.num_nodes)).toarray()#:D
            
    def epsilon_greedy(self,s_curr, q, epsilon):#exploraiton vs exploitation 
        potential_next_states = np.where(np.array(self.adjacent_mat[s_curr]) > 0)[0]
        if random.random() > epsilon:  
            q_of_next_states = q[s_curr][potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:  
            s_next = random.choice(potential_next_states)
        return s_next
