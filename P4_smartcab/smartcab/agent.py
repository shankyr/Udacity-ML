import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools
import pdb # for debugging
import os 
import os.path
import csv
import pandas as pd

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		self.valid_actions = [ None, 'forward', 'left', 'right'] 
		self.state = [] #we will use a list of tuples of states to map the indexes
		self.list_of_states = []
		self.next_waypoint = None
		self.number_of_states= (len(self.valid_actions)**4)*2 # 4 waypoints or actions * 2 light states * 4 oncoming  traffic states * 4 left traffic states  * 4 right  traffic states 
																#  next_waypoint, left, right and oncoming can take values 'None' , 'left', 'right' or 'forward'
		self.R = np.zeros(shape=[self.number_of_states, len(self.valid_actions)]) #we have 4 actions
		self.q = np.zeros(shape=[self.number_of_states, len(self.valid_actions)])
		self.policy = np.zeros([self.number_of_states, 1], dtype = int) # policy: what action will be taken in each state
		self.state_counter = 0
		self.alpha = 1. # learning rate		
		self.gamma = 0.0 #discount factor
		self.epsilon = 0.0
		self.data = pd.DataFrame({'alpha' : self.alpha, 'gamma' : self.gamma, 'epsilon' : self.epsilon,'successful': 0,'infractions' : 0, 'Q': [self.q], 'R': [self.R]})
		print self.data

		
	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		self.color = 'red'  # override colr
		self.state = []
		self.next_waypoint = None
		
	def write_success_to_csv(self, data):
		"""Write a line to the output CSV file"""
		# MODIFICATION
		output_file = open('output.csv', 'a') # append row to previous ones

		writer = csv.writer(output_file)#, delimiter='\n  ')
		# example row in the CSV file
		writer.writerow(data)
		output_file.close()
	
	def write_state_to_csv(self):
		"""Write a line to the output CSV file"""
		# MODIFICATION
		self.data['Q'] = [self.q]
		self.data['R'] = [self.R]
		try:
			df = pd.read_csv('outcome.csv')
			df= df.append(self.data) #self.data not in df so we append it
			#df_mask= pd.DataFrame({'alpha':df['alpha'].isin(self.data['alpha']),'gamma':df['gamma'].isin(self.data['gamma']), 'epsilon':df['epsilon'].isin(self.data['epsilon']),'successful': 0,'infractions' : 0, 'Q': [self.q], 'R': [self.R]})
			#pdb.set_trace()
			#mask = df[['alpha','gamma','epsilon']]==self.data[['alpha','gamma','epsilon']]
			#mask_update=['successful', 'infractions', 'Q', 'R']
			
			#if mask.values.all():#df[mask].isnull().values.any(): #if df[mask] contains nulls it means there is colission
				#pdb.set_trace()
				#df.iloc[df[mask].index].update(self.data.iloc[self.data[mask].index])
			#else:
				#df= df.append(self.data) #self.data not in df so we append 
			#df = pd.merge(df, self.data, how='right', on=['successful', 'infractions', 'Q', 'R'])
		except IOError:
			with open("outcome.csv", "w"):
			# now you have an empty file already
				df = self.data
				pass  # or write something to it already

		df.to_csv('outcome.csv',index=False)
		for filename, variable in zip(['list_of_states','reward','Q'] , [self.list_of_states, self.R, self.q]):
			output_file = open(filename+"_alpha_{}_gamma_{}_epsilon_{}.csv".format(self.alpha, self.gamma, self.epsilon), 'wb')# append row to previous ones
			writer = csv.writer(output_file, delimiter='\n')
			# example row in the CSV file
			writer.writerow(variable)#, self.q, self.R ))		
			
	def build_state(self, inputs, next_waypoint):
		'''Builds a state tuple'''
		return (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], next_waypoint)
	
	def store_states(self,this_state):
		'''Creates a list with the states that have appeared up to now (light,oncoming,right,left,next_waypoint)'''
		if not this_state in self.list_of_states:
			self.list_of_states.append(this_state)
			
	def get_reward(self, reward,t):
		if reward>=9.: #options are reward== 12. or reward == 9.5 or reward== 9. or reward ==10. : # success
			print 'SUCCESS!!!!!!!!!!!!!!!! \n \n'
			self.data['successful'] += 1
			self.write_success_to_csv([reward, t]) # save 
		elif reward == -1.:
			self.data['infractions']+= 1
	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state
		this_state =  self.build_state(inputs, self.next_waypoint)
		self.store_states(this_state)
		self.state.append(self.list_of_states.index(this_state))
	
		# TODO: Select action according to your policy
		action = self.select_action()
		
		# Execute action and get reward
		reward = self.env.act(self, action)
		# MODIFICATION https://discussions.udacity.com/t/efficiently-counting-the-number-of-times-car-reaches-destination-in-last-10-trials-out-of-100/174080/2
		self.get_reward(reward,t)
		self.R[self.state[-1]][self.valid_actions.index(action)] = reward
			#print self.R[range(len(self.list_of_states))][:]
		
		# TODO: Learn policy based on state, action, reward
		self.update_q(action,t+1.)
			#print self.q[range(len(self.list_of_states))][:]
		
		#we select the maximum argument from the list of possible actions for the state
		self.get_policy()

				#print self.policy[range(len(self.list_of_states))]
				
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

	def update_q(self, action, t):
		'''Update the Q matrix'''
		reward = self.R[self.state[-1]][self.valid_actions.index(action)] 
		next_inputs = self.env.sense(self)
		next_waypoint = self.planner.next_waypoint() 
		next_state = self.build_state(next_inputs, next_waypoint)
		temp = []
		self.store_states(next_state)
		temp= [self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions] 
		# First approach : Q_hat(s,a) = alpha(r+gamma*max(abs(Q_hat(s',a')-Q_hat(s,a))))
		self.alpha = 1./t
		self.q[self.state[-1]][self.valid_actions.index(action)]+= self.alpha* (reward +  self.gamma*(np.max(temp))-self.q[self.state[-1]][self.valid_actions.index(action)])
				# we iterate over the possible combinations of states and actions, since any future combination is possible 
	def select_action(self):
		if len(self.state)==1: # if we are in the first iteration 
		#1: random action
			action = random.choice(self.valid_actions)
		#1.b:  next waypoint can be a good direction to start, there is no policy yet
			action= self.next_waypoint
		else:
		#2_using policy to take an action
			probabilities = [1.-self.epsilon  , self.epsilon/float(len(self.valid_actions)) , self.epsilon/float(len(self.valid_actions)) , self.epsilon/float(len(self.valid_actions)), self.epsilon/float(len(self.valid_actions))]
			#probabilities = [1-e 0.25*e 0.25*e 0.25*e 0.25*e]
			these_actions = []						
			these_actions = [self.valid_actions[self.policy[self.state[-2]]]]  + self.valid_actions
			action =  np.random.choice(these_actions, 1, p= probabilities)[0]#we access the state previous to the present one in order to choose the action to take 
		return action
		
	def get_policy(self):
		next_inputs = self.env.sense(self)
		next_waypoint = self.planner.next_waypoint() 
		next_state = self.build_state(next_inputs, next_waypoint)
		self.policy[self.state[-1]]= np.argmax([self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions]) 
		return self.policy[self.state[-1]]
		


def remove_file(filename):
	try:
		os.remove(filename)
	except OSError:
		pass
		
def run():
	"""Run the agent for a finite number of trials."""
	#pdb.set_trace()
	
	#os.remove(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), 'output.csv')) #reset statistics
	#os.remove(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), 'simulation.csv')) #reset statistics 
	list_of_files = ['simulation.csv', 'output.csv', 'list_of_states.csv','reward.csv','Q.csv']
	for filename in list_of_files:
		remove_file(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), filename))
	# Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
	sim = Simulator(e, update_delay=0., display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=100)#run for a specified number of trials
	a.write_state_to_csv()
	
	# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
	
	#pdb.set_trace()



if __name__ == '__main__':
    run()
