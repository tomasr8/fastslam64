import numpy as np
import pycuda.driver as cuda
import math
from config_ros import config
from cuda.fastslam import load_cuda_modules
from lib.common import CUDAMemory, resample, rescale, get_pose_estimate
from lib.particle3 import FlatParticle


class Slam:
    def __init__(self,seed=None,start_position= None):
    # seed init
        if seed is None:
            self.seed = config.SEED
        else:
            self.seed = seed
        np.random.seed(self.seed)

        assert config.THREADS <= 1024 # cannot run more in a single block
        assert config.N >= config.THREADS
        assert config.N % config.THREADS == 0
        self.measurements = None
        self.odometry= None

        theta = self.get_heading(start_position[1])
        theta = self.pi_2_pi(theta)
        start_position = [start_position[0][0],start_position[0][1],theta]

        #particles init
        self.particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, start_position, sigma=0.2)

        self.cuda_modules = load_cuda_modules(
                THREADS=config.THREADS,
                PARTICLE_SIZE=config.PARTICLE_SIZE,
                N_PARTICLES=config.N
            )

        self.memory = CUDAMemory(config)
        self.weights = np.zeros(config.N, dtype=np.float64)

        cuda.memcpy_htod(self.memory.cov, config.sensor.COVARIANCE)
        cuda.memcpy_htod(self.memory.particles, self.particles)
        self.__cuda_init_rng__()

    def set_measurements(self,measurements):
        self.measurements = np.array([self.__xy2rb__(self.odometry, m) for m in measurements], dtype=np.float64)

        self.__cuda_reset_weights__()
      
        cuda.memcpy_htod(self.memory.measurements, self.measurements)
        self.__cuda_predict_from_imu__()
        self.__cuda_update__()
        rescale(self.cuda_modules, config, self.memory)
        estimate = get_pose_estimate(self.cuda_modules, config, self.memory)
        best = np.argmax(FlatParticle.w(self.particles))
        cuda.memcpy_dtoh(self.particles, self.memory.particles)
        landmarks = FlatParticle.get_landmarks(self.particles, best)
        self.__cuda_get_weights__()
        cuda.memcpy_dtoh(self.weights, self.memory.weights)
        neff = FlatParticle.neff(self.weights)
        if neff < 0.6*config.N:
            resample(self.cuda_modules, config, self.weights, self.memory, 0.5)
        return estimate,landmarks
    
    # odometry : ((x,y),(x,y,z,w)) (position,orientation)
    def set_odometry(self,odometry:tuple):
        theta = self.get_heading(odometry[1])
        theta = self.pi_2_pi(theta)
        self.odometry = [odometry[0][0],odometry[0][1],theta]

    def get_heading(self, o):
        roll = np.arctan2(
            2.0*(o[0]*o[1] + o[3]*o[2]),
            o[3]*o[3] + o[0]*o[0] - o[1]*o[1] - o[2]*o[2]
        )
        return roll + np.pi/2

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def __cuda_init_rng__(self):
        self.cuda_modules["predict"].get_function("init_rng")(
            np.int32(self.seed), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

    def __cuda_reset_weights__(self):
        self.cuda_modules["resample"].get_function("reset_weights")(
            self.memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

    def __cuda_predict_from_imu__(self):
        self.cuda_modules["predict"].get_function("predict_from_imu")(
            self.memory.particles,
            np.float64(self.odometry[0]), np.float64(self.odometry[1]), np.float64(self.odometry[2]),
            np.float64(config.ODOMETRY_VARIANCE[0] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[1] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[2] ** 0.5),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

    def __cuda_update__(self):
        block_size = config.N if config.N < 32 else 32
        self.cuda_modules["update"].get_function("update")(
            self.memory.particles, np.int32(1),
            self.memory.scratchpad, np.int32(self.memory.scratchpad_block_size),
            self.memory.measurements,
            np.int32(config.N), np.int32(len(self.measurements)),
            self.memory.cov, np.float64(config.THRESHOLD),
            np.float64(config.sensor.RANGE), np.float64(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(block_size, 1, 1), grid=(config.N//block_size, 1, 1)
        )
    
    def __cuda_get_weights__(self):
        self.cuda_modules["weights_and_mean"].get_function("get_weights")(
            self.memory.particles, self.memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

    def __xy2rb__(self,pose, landmark):
        color = landmark[2]
        position = pose[:2]
        vector_to_landmark = np.array(landmark[:2] - position, dtype=np.float64)

        r = np.linalg.norm(vector_to_landmark)
        b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]
        b = self.__wrap_angle__(b)

        return r, b, color    

    def __wrap_angle__(self,angle):
        return np.arctan2(np.sin(angle), np.cos(angle))