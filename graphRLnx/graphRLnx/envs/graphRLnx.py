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
        self.start_state = 1
        self.aim_state = [4,11]
        self.fire = [2,3]
        
        self.current_state = self.start_state
        self.reward = 0
                
        self.time_step = 0
        self.done = False
        
        self.edges = [(0,1), (1,0), (0,2), (2,0), (0,12), (12,0), (1,2), (2,1), (1,8), (8,1), (1,12), (12,1), 
         (1,13), (13,1), (2,3), (3,2), (2,8), (8,2), (2,12), (12,2), (3,4), (4,3), (3,8), (8,3), 
         (3,13), (13,3), (4,5), (5,4), (4,6), (6,4), (4,7), (7,4), (4,9), (9,4), (4,10), (10,4), 
         (4,14), (14,4), (5,6), (6,5), (5,7), (7,5), (5,10), (10,5), (6,7), (7,6), (6,10), (10,6), 
         (6,11), (11,6), (7,11), (11,7),(8,12), (12,8), (8,13), (13,8), (9,10), (10,9), (9,13), 
         (13,9), (9,14), (14,9), (9,16), (16,9), (10,11), (11,10), (10,14), (14,10), (10,15), (15,10), 
         (11,15), (11,18), (12,13), (13,12), (12,16), (16,12), (13,14), (14,13), (13,16), (16,13), 
         (14,15), (15,14), (14,16), (16,14), (14,17), (17,14), (15,17), (17,15), (15,18), (18,15), 
         (16,17), (17,16), (17,18), (18,17)]
        
        G = nx.Graph()
        G.add_edges_from(self.edges)
        pos = nx.spring_layout(G)
                
        #Drawn graph
        color_map = []
        for node in G:
            if node in self.fire:
                color_map.append('red')
            elif node in self.aim_state:
                color_map.append('green')
            elif node == self.start_state:
                color_map.append('pink')
            else:
                color_map.append('blue')
                            
        nx.draw_networkx_nodes(G, pos, node_color=color_map)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
                        
        plt.show()                
                        
        self.graph = G
        self.adjacent_mat = nx.adjacency_matrix(self.graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(self.graph, nodelist=range(self.num_nodes)).toarray()#:D

        
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))  # num_states * num_actions
        
        self.reset(self.start_state, self.aim_state, self.fire, self.edges)

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

    def reset(self, start_state, aim_state, fire, edges):
        self.start_state = start_state
        self.aim_state = aim_state
        self.fire = fire
        
        self.current_state = self.start_state
        self.reward = 0
                
        self.time_step = 0
        self.done = False
        
        self.edges = edges
        G = nx.Graph()
        G.add_edges_from(self.edges)
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
