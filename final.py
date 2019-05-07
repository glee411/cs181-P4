# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt
from math import floor

from SwingyMonkey import SwingyMonkey

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
    	self.last_state  = None
    	self.last_action = None
    	self.last_reward = None
    	self.hsize = 600 # don't change
    	self.vsize = 400 # don't change
    	self.size_hg = 100
    	self.size_vd = 50
    	self.size_vg = 100
    	self.size_vel = 15
    	self.hbins = 20 # int(floor(self.hsize / self.size_hg)*2) # number of states in horizontal distance feature
    	self.vbins = 20 # number of states in vertical distance feature
    	self.vgbins = 20 # int(floor(self.vsize / self.size_vd)*2) # number of states in vertical gap feature
    	self.velbins = 20 # 4 #int(floor(50 / self.size_vel)*2) # number of states in velocity feature
    	self.Q = np.zeros((self.hbins,self.vgbins,self.velbins,2,2)) # number states x number actions
    	self.stateCounts = np.zeros((self.hbins,self.vgbins,self.velbins,2)) # state space to track how many times you've visited a state
    	self.scmax = 0 # counter for state with maximum number of visits
    	# self.rate = .2 # learning rate
    	self.discount = .99 # discount rate
    	self.eps = 0.01 # episilon
    	self.g = None # TODO: adjust gravity later
    	self.iteration = 0 # ticks
    	self.vertvals = []
    	self.horizvals = []
    	self.vertgapvals = []
    	self.velvals = []

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = None # adjust gravity

    def __make_discrete(self, state):
    	# split distance to tree horiz (hbins)
    	horiz_dist = state["tree"]["dist"]
    	self.horizvals.append(horiz_dist)
    	horiz_to_tree = int(horiz_dist // 120)
    	# horiz_to_tree = floor(horiz_dist/self.size_hg) + floor(self.hsize / self.size_hg)

    	# split distance to top (vbins, always positive)
    	dist_val = state["monkey"]["top"]
    	self.vertvals.append(dist_val)
    	vert_to_top = floor(dist_val/self.size_vd)

        # split distance to tree vert (vgbins)
    	vert_dist = state["tree"]['top']-state["monkey"]["top"]
    	self.vertgapvals.append(vert_dist)
    	vert_to_tree = int(vert_dist // 60)
    	# vert_to_tree = floor(vert_dist/self.size_vd) + floor(self.vsize / self.size_vd)

    	# split velocity (velbins)
    	cur_vel = state["monkey"]["vel"]
    	self.velvals.append(cur_vel)
    	if cur_vel <= 15: # bc of poisson
    		velocity = 0
    	else:
    		velocity = 1

    	return horiz_to_tree,vert_to_tree, velocity


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # if we've done this once then update
        if self.last_state != None:
        	if self.g == None:
        		self.g = int(-1*state["monkey"]["vel"])
        		if self.g == 4:
        			self.g = 0 # 4 is stored as 0
        		else:
        			self.g = 1
        	h,vg,vel = self.__make_discrete(state)
        	la = int(self.last_action)
        	cur_Q = self.Q[h][vg][vel][self.g][la]
        	self.stateCounts[h][vg][vel][self.g] += 1
        	experience = self.stateCounts[h][vg][vel][self.g]
        	if self.stateCounts[h][vg][vel][self.g] > self.scmax:
        		self.scmax = self.stateCounts[h][vg][vel][self.g]
        	calc_max = max(self.Q[h][vg][vel][self.g])
        	new_Q = cur_Q + rate *(self.last_reward + self.discount*calc_max - cur_Q)
        	self.Q[h][vg][vel][self.g][la] = new_Q
        	if(self.Q[h][vg][vel][self.g][0] >= self.Q[h][vg][vel][self.g][1]):
        		self.last_action = 0
        	else:
        		self.last_action = 1
        	if npr.rand() < (self.eps/experience):
        		self.last_action = 1-self.last_action
        	self.last_state  = state
        else:
        	new_action = int(npr.rand() < self.eps)
        	new_state  = state
        	self.last_action = new_action
        	self.last_state  = new_state

        self.iteration += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, vertvals, horizvals, vertgapvals, velvals, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	#hist = []
	vertvals = []
	horizvals = []
	vertgapvals = []
	velvals = []
	rate = 0
	lrs = [0.05, 0.1, 0.2]
	res_dict = []

	# Run games.
	for lr in lrs:
		rate = lr
		res = []
		for i in range(10):
			hist = []
			run_games(agent, hist,  vertvals, horizvals, vertgapvals, velvals, 50, 1)
			res.append(np.max(hist))
		res_dict.append(res)
	name = "lr.csv"
	np.savetxt(name, res_dict, delimiter=",")





	# run_games(agent, hist,  vertvals, horizvals, vertgapvals, velvals, 75, 10)

	# # Save history.
	# np.save('hist',np.array(hist))

	# # print history
	# plt.plot(hist)
	# plt.show()

	# # print history hist
	# plt.hist(hist)
	# plt.show()
