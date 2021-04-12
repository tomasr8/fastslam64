import math
import numpy as np

class Vehicle(object):
    def __init__(self, position, movement_variance, dt):
        self.position = position
        self.movement_variance = movement_variance
        self.dt = dt
        self.random = np.random.RandomState(seed=4)

    def move_noisy(self, u):
        '''Stochastically moves the vehicle based on the control input and noise
        '''
        x, y, theta = self.position

        if u[0] == 0.0 and u[1] == 0.0:
            return

        theta += (u[0] * self.dt) + self.random.normal(0, self.movement_variance[0])
        theta %= (2*math.pi)

        dist = (u[1] * self.dt) + self.random.normal(0, self.movement_variance[1])
        x += np.cos(theta) * dist
        y += np.sin(theta) * dist

        self.position = np.array([x, y, theta]).astype(np.float32)