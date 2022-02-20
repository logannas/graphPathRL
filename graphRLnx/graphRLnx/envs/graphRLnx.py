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
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    metadata = {'render.modes': ['human', 'graph', 'interactive']}

    def __init__(self, network_size=5, input_nodes=3):
        self.start_state = 0
        self.aim_state = [4]
        
        self.current_state = self.start_state
        self.reward = 0
        
        self.network_size = network_size
        
        self.time_step = 0
        self.done = False
        
        edges = [(0,1), (0,3), (0,2), (1,0), (1,2), (1,3), (2,0), (1,2), (2,4), (3,0), (3,1), (3,4)]
        graph = nx.Graph()
        graph.add_edges_from(edges)
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        
        self.graph = graph
        self.adjacent_mat = nx.adjacency_matrix(graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(graph, nodelist=range(self.num_nodes)).toarray()#:D

        pl.show()
        
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))  # num_states * num_actions
        
        self.reset(self.start_state, self.aim_state)


    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'human':
            nx.draw(self.graph, with_labels=True, font_weight='bold')
            plt.show()

    def step(self, action):
        epsilon=0.05 
        alpha=0.1
        gamma=0.8
        
        next_state = self.epsilon_greedy(self.current_state, self.q_table, epsilon=epsilon)
        s_next_next = self.epsilon_greedy(next_state, self.q_table, epsilon=-0.2)  # epsilon<0, greedy policy
        # update q_table
        reward = -self.adjacent_mat[self.current_state][next_state]
        delta = reward + gamma * self.q_table[next_state, s_next_next] - self.q_table[self.current_state, next_state]
        
        self.q_table[self.current_state, next_state] = self.q_table[self.current_state, next_state] + alpha * delta
        # update current state
        self.current_state = next_state 

        if self.current_state in self.aim_state:
            self.done = True
        return self.current_state, reward, self.done, {"time_step": self.time_step}

    def reset(self, start_state, aim_state):
        self.start_state = start_state
        self.aim_state = aim_state
        
        self.current_state = self.start_state
        self.reward = 0
                
        self.time_step = 0
        self.done = False
        
        edges = [(0,1), (0,3), (0,2), (1,0), (1,2), (1,3), (2,0), (1,2), (2,4), (3,0), (3,1), (3,4)]
        graph = nx.Graph()
        graph.add_edges_from(edges)
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        
        self.graph = graph
        self.adjacent_mat = nx.adjacency_matrix(graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(graph, nodelist=range(self.num_nodes)).toarray()#:D
            
    def epsilon_greedy(self,s_curr, q, epsilon):#exploraiton vs exploitation 
        potential_next_states = np.where(np.array(self.adjacent_mat[s_curr]) > 0)[0]
        if random.random() > epsilon:  
            q_of_next_states = q[s_curr][potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:  
            s_next = random.choice(potential_next_states)
        return s_next
