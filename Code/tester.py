from flat_game import carmunk
import numpy as np
import random


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(0, 3)

    def update(self, next_state, reward):
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

    def learn(self,nb_episodes,filename,load_model,model_path):

        # if we want to load some trained parameters
        if load_model:
            self.agent.W = self.load_trained_model(model_path)


        # Initial state
        _,state = self.carmunk.frame_step(random.randint(0,3))
        # storing distance made by the car
        car_distance = 0

        distance_training = []
        distance_measure = []

        episode = 1

        while episode <= nb_episodes:
            car_distance += 1
            reward,new_state = self.carmunk.frame_step(self.agent.act(state))
            
            self.agent.update(state,new_state,reward)

            state = new_state

            if reward == -500:
                if nb_episodes - episode < 50:
                    distance_measure.append(car_distance)
                else:
                    distance_training.append(car_distance)
                print("episode : %d distance_made : %d" % (episode, car_distance))
                
                car_distance = 0
                episode += 1

            if episode%1000 == 0:
                #np.savetxt('Qmat_'+str(episode)+'.txt',self.agent.Qmat,delimiter = ',')
                np.savetxt(filename+str(episode)+'_episodes.txt',self.agent.W,delimiter=',')

        print(distance_measure)
        return np.mean(np.array(distance_measure))


    def load_trained_model(self,path):
    	trained_model = np.loadtxt(path,delimiter=',')
    	return trained_model



if __name__ == '__main__':
	
	nb_episodes = 1000
	gamma = 0.1
	alpha=0.1
	epsilon = 0.1
	p = 30
	k= 10
	n_arm = 3
	max_arm = 39
	nb_actions = 3
	state_dims = [40,40,40]

	path = 'Qmat_5000.txt'
	#agent = RandomAgent()
	agent = Qlearning_discrete(epsilon,alpha,gamma,nb_actions,state_dims)
	#agent = Qlearning_cont_3Sensors(epsilon,alpha,gamma,n_arm,max_arm,nb_actions)
	#agent = Qlearning_XY3Sensors(epsilon,alpha,gamma,p,k,n_arm,max_arm,nb_actions)
	a = Tester(agent)
	b = a.learn(nb_episodes,'Qmat_',True,path)

	print(b)




