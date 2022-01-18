#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define N_PARTICLES <<N_PARTICLES>>
#define PARTICLE_SIZE <<PARTICLE_SIZE>>
#define THREADS <<THREADS>>

__device__ double* get_particle(double *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

/*
 * Copies particles in place given by the ancestor vector
 */
 __global__ void copy_inplace(
    double *particles, int *ancestors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i == ancestors[i]) {
        return;
    }

    double *source = get_particle(particles, ancestors[i]);
    double *dest = get_particle(particles, i);

    int max_landmarks = (int)source[4];
    int n_landmarks = (int)source[5];

    dest[0] = source[0];
    dest[1] = source[1];
    dest[2] = source[2];
    dest[3] = source[3];
    dest[4] = source[4];
    dest[5] = source[5];

    for(int k = 0; k < n_landmarks; k++) {
        dest[6+2*k] = source[6+2*k];
        dest[6+2*k+1] = source[6+2*k+1];

        dest[6+2*max_landmarks+4*k] = source[6+2*max_landmarks+4*k];
        dest[6+2*max_landmarks+4*k+1] = source[6+2*max_landmarks+4*k+1];
        dest[6+2*max_landmarks+4*k+2] = source[6+2*max_landmarks+4*k+2];
        dest[6+2*max_landmarks+4*k+3] = source[6+2*max_landmarks+4*k+3];

        dest[6+6*max_landmarks+k] = source[6+6*max_landmarks+k];
        
        dest[6+7*max_landmarks+k] = source[6+7*max_landmarks+k];
    }
}


 __global__ void copy_inplace_coalesced(
    double *particles, int *ancestors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for(int k = 0; k < N_PARTICLES; k++) {
        double *source = get_particle(particles, ancestors[i]);
        double *dest = get_particle(particles, i);

        dest[i] = source[i];
    }
}


__global__ void reset_weights(double *particles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    double *particle = get_particle(particles, i);
    particle[3] = 1.0/N_PARTICLES;
}


__global__ void systematic_resample(double *weights, double *cumsum, double rand, int *ancestors) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int left = ceil((cumsum[i]*N_PARTICLES - weights[i]*N_PARTICLES) - rand);
    int right = ceil((cumsum[i]*N_PARTICLES) - rand);

    for(int j = left; j < right; j++) {
        ancestors[j] = i;
    }
}

/*
 * Calculates neff.
 * Needs to run in a single block.
 */
 __global__ void get_neff(double *particles, double *neff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    double square_sum = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        double *particle = get_particle(particles, i);
        square_sum += (double)particle[3] * (double)particle[3];
    }

    __shared__ double r_square_sum[THREADS];
    r_square_sum[idx] = square_sum;

    __syncthreads();

    for (int size = THREADS/2; size > 0; size /= 2) {
        if (idx < size) {
            r_square_sum[idx] += r_square_sum[idx + size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *neff = 1.0/r_square_sum[0];
    }
}
