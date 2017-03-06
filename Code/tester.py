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

class Tester():

    def __init__(self, agent):
        self.carmunk = carmunk.GameState()
        self.agent = agent

    def learn(self,nb_frames):
        # Initial state
        _,state = self.carmunk.frame_step(random.randint(0,3))
        car_distance = 0
        distance = []

        frame = 1
        while frame <= nb_frames:
            car_distance += 1
            reward,new_state = self.carmunk.frame_step(self.agent.act())
            self.agent.update(new_state,reward)

            state = new_state

            if reward == -500:
                distance.append(car_distance)
                print("t: %d distance_made : %d" % (frame, car_distance))
                car_distance = 0

            frame += 1
        return distance


if __name__ == '__main__':
	nb_frames = 10000
	agent = RandomAgent()
	a = Tester(agent)
	b = a.learn(nb_frames)

	print(b)




