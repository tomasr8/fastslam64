#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ double* get_particle(double *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

// Manual extern "C" to stop name mangling shenanigans
// Otherwise doesn't compile because curand complains
extern "C" {

// Based on https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel    
// Each thread has a random state
__device__ curandState_t* states[<<N_PARTICLES>>];


// This function is only called once to initialize the rngs.
__global__ void init_rng(int seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, i, 0, s);
    states[i] = s;
}


__global__ void predict_from_imu(double *particles,
    double x, double y, double theta, double sigma_x, double sigma_y, double sigma_theta) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);
    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[0] = x + sigma_x * curand_normal(states[i]);
    particle[1] = y + sigma_y * curand_normal(states[i]);
    particle[2] = theta + sigma_theta * curand_normal(states[i]);
}

// Moves particles based on the control input and movement model.
__global__ void predict_from_model(double *particles, double ua, double ub, double sigma_a, double sigma_b, double dt) {
    if(ua == 0.0 && ub == 0.0) {
        return;
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);

    ua += sigma_a * curand_normal(states[i]);
    ub += sigma_b * curand_normal(states[i]);

    double angle = particle[2];

    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[2] += (ua * dt);
    particle[2] = fmod(particle[2], (double)(2*M_PI));
    // particle[2] = atan2(sin(particle[2]), cos(particle[2]));


    double dist = (ub * dt);
    particle[0] += cos(angle) * dist;
    particle[1] += sin(angle) * dist;
}

// Moves particles based on the control input and movement model.
__global__ void predict_from_fsonline_model(double *particles, double ua, double ub, double uc, double sigma_a, double sigma_b, double dt) {
    if(ua == 0.0 && ub == 0.0 && uc == 0.0) {
        return;
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);

    ua += sigma_a * curand_normal(states[i]);
    ub += sigma_b * curand_normal(states[i]);
    uc += sigma_a * curand_normal(states[i]);

    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[2] += (ua * dt);
    // particle[2] = fmod(particle[2], (double)(2*M_PI));
    particle[2] = fmod(particle[2], (double)(2*M_PI));
    // particle[2] = atan2(sin(particle[2]), cos(particle[2]));

    double dist = (ub * dt);
    particle[0] += cos(particle[2]) * dist;
    particle[1] += sin(particle[2]) * dist;

    particle[2] += (uc * dt);
    particle[2] = fmod(particle[2], (double)(2*M_PI));
}


}

