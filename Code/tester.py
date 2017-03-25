from flat_game import carmunk
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self,state):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(0, 3)

    def update(self, state,next_state, reward):
        """
        Update your internal state
        """
        pass




class Qlearning_discrete():
    def __init__(self,epsilon,alpha,gamma,nb_actions,state_dims):
        self.nb_actions = nb_actions
        self.state_dims = state_dims
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.W = np.zeros((np.prod(self.state_dims), self.nb_actions))

    def ravel(self,state):
        id = state[0]
        for s, dim in zip(state[1:], self.state_dims[1:]):
            id = id * dim + s
        return int(id)

    def act(self,state):

        if np.random.rand() < self.epsilon:
            self.current_action = np.random.randint(0,3)
        else:
            
            self.current_action = np.argmax(self.W[self.ravel(state[2:])])
        return self.current_action

    def update(self,state,next_state,reward):
        state_id = self.ravel(state[2:])
        old = self.W[state_id,self.current_action]
        self.W[state_id,self.current_action] = old + self.alpha*(reward + self.gamma*max(self.W[self.ravel(next_state[2:])]) - old)
            

class Qlearning_cont_3Sensors():

    def __init__(self,epsilon,alpha,gamma,n_arm,max_arm,nb_actions):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon 
        self.n_arm = n_arm
        self.max_arm = max_arm
        self.nb_actions = nb_actions
        self.nrows = self.n_arm 

        # matrix of weights
        self.W = np.array(np.zeros(self.nrows*self.nb_actions).reshape((self.nrows,self.nb_actions)))

        #self.W = np.array(np.zeros(self.nrows2*self.nb_actions).reshape((self.nrows2,self.nb_actions)))

    def act(self,state):
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(0,3)
        else:
            self.action = np.argmax(np.dot(np.transpose(self.W),np.array(state[2:])/self.max_arm))
        return self.action

    def update(self,state,next_state,reward):
    
        
        old_W = self.W


        old_Q = float(np.dot(np.transpose(old_W[:,self.action]),np.array(state[2:])/self.max_arm))
        new_Q = float(np.max(np.dot(np.transpose(old_W),np.array(next_state[2:])/max_arm)))
        temp = reward + self.gamma*new_Q - old_Q
        
  

        self.W[:,self.action] = old_W[:,self.action] + ((self.alpha*temp)*np.array(state[2:])/max_arm).reshape((self.nrows,))







class Qlearning_XY3Sensors():

    def __init__(self,epsilon,alpha,gamma,p,k,n_arm,max_arm,nb_actions):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon 
        self.p = p 
        self.k = k
        self.n_arm = n_arm
        self.max_arm = max_arm
        self.nb_actions = nb_actions
        self.nrows = (self.p+1)*(self.k+1) + self.n_arm 
         # matrix of discrete state
        self.S = np.zeros((self.p+1,self.k+1,2))
        
        for i in range(self.p +1):
            self.S[i,:,0] =  i*(1000/self.p)
        for j in range(self.k+1):
            self.S[:,j,1] = j*(700/self.k)
        
        # matrix of weights
        self.W = np.array(np.zeros(self.nrows*self.nb_actions).reshape((self.nrows,self.nb_actions)))
        # matrix phi
        self.phi = np.array(np.random.rand(self.nrows)).reshape((self.nrows,1))



    def act(self,state):
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(0,3)
        else:
            self.action = np.argmax(np.dot(np.transpose(self.W),self.phi))
        return self.action



    def update(self,state,next_state,reward):
         
        
        #compute phi
        old_phi = self.phi
        old_W = self.W
        # update phi 
        
        loc_features = np.reshape(np.apply_along_axis(np.prod,2,np.exp(-1*np.power(np.add(next_state[:2],-1*self.S),2))),(self.nrows-self.n_arm,1)) # 
        self.phi = np.concatenate((loc_features,(np.array(next_state[2:])/self.max_arm).reshape(self.n_arm,1)),axis=0)
       

        old_Q = float(np.dot(np.transpose(old_W[:,self.action]),old_phi))
        
        new_Q = float(np.max(np.dot(np.transpose(old_W),self.phi)))
        temp = reward + self.gamma*new_Q - old_Q
        

        self.W[:,self.action] = old_W[:,self.action] + ((self.alpha*temp)*old_phi).reshape((self.nrows,))


class Tester():

    def __init__(self, agent):
        self.carmunk = carmunk.GameState()
        self.agent = agent

    def learn(self,nb_episodes,path=None,save_coef =False,load_model=False,model_path=None):

        # if we want to load some trained parameters
        if load_model:
            self.agent.W = self.load_trained_model(model_path)


        # Initial state
        _,state = self.carmunk.frame_step(random.randint(0,3))
        # storing distance made by the car
        car_distance = 0
        distance_measure = []

        episode = 1

        while episode <= nb_episodes:
            car_distance += 1
            reward,new_state = self.carmunk.frame_step(self.agent.act(state))
            
            self.agent.update(state,new_state,reward)

            state = new_state

            if reward == -500: # means collision 

                
                distance_measure.append(car_distance)
                print("episode : %d distance_made : %d" % (episode, car_distance))
                car_distance = 0
                episode += 1

            
            if save_coef:

                if episode%2000 == 0:
               	    os.makedirs(path, exist_ok=True)
                    np.savetxt(path+'Qmat_'+str(episode)+'_episodes.txt',self.agent.W,delimiter=',')

        
        return distance_measure,np.mean(np.array(distance_measure))


    def load_trained_model(self,path):
        trained_model = np.loadtxt(path,delimiter=',')
        return trained_model

    def play_model(self,nb_episodes,model_path):

        self.agent.W = self.load_trained_model(model_path)
        # Initial state
        _,state = self.carmunk.frame_step(random.randint(0,3))
        # storing distance made by the car
        car_distance = 0

        distance = []
        episode = 1

        while episode <= nb_episodes:
            car_distance += 1
            reward,new_state = self.carmunk.frame_step(self.agent.act(state))

            state = new_state
            # the agent does not learn now, only choose action depending on the learned parameters (loaded model)

            if reward == -500:
                
                    distance.append(car_distance)
                    print("episode : %d distance_made : %d" % (episode, car_distance))
                    car_distance = 0
                    episode += 1

        print('Average distance made on the ' + str(nb_episodes) + ' episodes : ' + str(np.mean(np.array(distance))))
        return np.mean(np.array(distance))






if __name__ == '__main__':
	
	nb_episodes = 2000
	gamma = 0.5
	alpha=0.5
	epsilon = 0.1
	p = 30
	k= 10
	n_arm = 3
	max_arm = 39
	nb_actions = 3
	state_dims = [40,40,40]


	if False:
		""" Compare random agent, and the 3 q learning agent with the same parameters 
			We are training over 5000 episodes and recording the distances made for each episode
			The second step will be to construct a 50 episodes moving average of the distances for 
			each agent """

		# Random agent 
		agents = {'random':RandomAgent(),
				'discrete_sensors':Qlearning_discrete(epsilon,alpha,gamma,nb_actions,state_dims),
				'continuous_sensors':Qlearning_cont_3Sensors(epsilon,alpha,gamma,n_arm,max_arm,nb_actions),
				'discrete_XY_continuous_sensors':Qlearning_XY3Sensors(epsilon,alpha,gamma,p,k,n_arm,max_arm,nb_actions)}

		res_distances = {}

		for agent in agents:
			print(agent)
			print('\n')
			test = Tester(agents[agent])
			distances, mean_distance = test.learn(3000)
			res_distances[agent] = distances

		df_res = pd.DataFrame.from_dict(res_distances,orient='index').sort_index()
		df_res.to_csv('Comparing_agents/distances_made.txt')

	if True:

		# Load distances from previous training 
		data_distances = pd.read_csv('Comparing_agents/distances_made.txt')
		res_mean = {}
		for i in range(len(data_distances)):

			agent = data_distances.iloc[i,0]
			distances = data_distances.iloc[i,1:].tolist()

			res_temp = []

			for j in range(len(distances)-300):
				res_temp.append(sum(distances[j:j+300])/300)

			print('.'),

			res_mean[agent] = res_temp


		# Plot the results 

		# list for legend
		legend = []
		for agent in res_mean:

			plt.plot(res_mean[agent])
			legend.append(str(agent))
		
		plt.title('Mean distance for moving average over 300 episodes')
		plt.ylabel('Mean distance')
		plt.legend(legend)
		plt.show()













	path = 'Qmat_5000.txt'
	#agent = RandomAgent()
	#agent = Qlearning_discrete(epsilon,alpha,gamma,nb_actions,state_dims)
	#agent = Qlearning_cont_3Sensors(epsilon,alpha,gamma,n_arm,max_arm,nb_actions)
	#agent = Qlearning_XY3Sensors(epsilon,alpha,gamma,p,k,n_arm,max_arm,nb_actions)
	#a = Tester(agent)
	#b = a.learn(nb_episodes,'Qmat_',True,path)
	#Tester(agent).play_model(1000,'Discrete3Sensors/Qmat_5000.txt')
	


	if False:

		""" Train each discrete agent with a pair of parameters (alpha,gamma)
			We stop training after 2000 episodes and save the learned matrix
			CAREFUL : NEED MORE THAN 5 HOURS OF TRAINING FOR ALL THESE MODELS (25 different models)"""

		for alpha in [0.9]:
			for gamma in [0.1,0.3,0.5,0.7,0.9]:

				agent = Qlearning_discrete(epsilon,alpha,gamma,nb_actions,state_dims)
				test = Tester(agent)
				distances,mean_distance = test.learn(nb_episodes=nb_episodes,path='Discrete3Sensors/alpha_'+str(alpha)+'_gamma_'+str(gamma),
					save_coef=True,load_model=False)

	if False: 

		""" We load the previous learned model and we measure the mean distance made 
			over 200 episodes for each trained model"""

		# folder containing all our agent 
		output = [d for d in os.listdir('Discrete3Sensors') if os.path.isdir(os.path.join('Discrete3Sensors',d))]

		res = {}
		
		for directory in output:
			# initialize the qlearning discrete agent with random values, we'll not use them because we will only play our trained model 

			agent = Qlearning_discrete(0.1,0.1,0.1,nb_actions,state_dims)
			test = Tester(agent)
			player = test.play_model(200,'Discrete3Sensors/'+directory+'/Qmat_2000_episodes.txt')

			res[directory] = player
		
		df_res = pd.DataFrame.from_dict(res,orient='index').sort_index()
		df_res.to_csv('Results_Parameters_Discrete/performances.txt')






