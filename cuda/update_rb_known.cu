#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define M_PI 3.14159265359
#define MIN(a,b) (((a)<(b))?(a):(b))


typedef struct 
{
    double (*measurements)[3];
    int n_measurements;
    double *measurement_cov;
} landmark_measurements;

__device__ double mod_angle(double angle) {
    return fmod(angle, (double)(2*M_PI));
}


__device__ void to_coords(double *particle, double *in, double *out) {
    double x = particle[0];
    double y = particle[1];
    double theta = particle[2];

    double range = in[0];
    double bearing = in[1];

    out[0] = x + range * cos(bearing + theta);
    out[1] = y + range * sin(bearing + theta);
}

__device__ double* get_particle(double *particles, int i) {
    int max_landmarks = (int)particles[4];
    return (particles + (6 + 7*max_landmarks)*i);
}

__device__ double* get_mean(double *particle, int i)
{
    return (particle + 6 + 2*i);
}

__device__ double* get_cov(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 2*max_landmarks + 4*i);
}

__device__ double* get_landmark_id(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 6*max_landmarks + i);
}

__device__ void set_landmark_id(double *particle, int i, double id)
{
    int max_landmarks = (int)particle[4];
    double *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] = id;
}

__device__ int get_n_landmarks(double *particle)
{
    return (int)particle[5];
}

__device__ void add_landmark(double *particle, double mean[2], double *cov, double id)
{
    int n_landmarks = (int)particle[5];
    particle[5] = (double)(n_landmarks + 1);

    double *new_mean = get_mean(particle, n_landmarks);
    double *new_cov = get_cov(particle, n_landmarks);
    double *new_id = get_landmark_id(particle, n_landmarks);

    new_mean[0] = mean[0];
    new_mean[1] = mean[1];

    new_cov[0] = cov[0];
    new_cov[1] = cov[1];
    new_cov[2] = cov[2];
    new_cov[3] = cov[3];

    new_id[0] = id;
}

__device__ void vecmul(double *A, double *u, double *v)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double e = u[0];
    double f = v[1];

    v[0] = a*e + b*f;
    v[1] = c*e + d*f;
}

__device__ void matmul(double *A, double *B, double *C)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double e = B[0];
    double f = B[1];
    double g = B[2];
    double h = B[3];

    C[0] = a*e + b*g;
    C[1] = a*f + b*h;
    C[2] = c*e + d*g;
    C[3] = c*f + d*h;
}

__device__ void matmul_jacobian(double *H, double *E, double *R, double *S)
{
    double a = H[0];
    double b = H[1];
    double c = H[2];
    double d = H[3];

    double Ht[] = {
        a, c,
        b, d
    };

    matmul(H, E, S);
    matmul(S, Ht, S);

    S[0] += R[0];
    S[1] += R[1];
    S[2] += R[2];
    S[3] += R[3];
}

__device__ void pinv(double *A, double *B)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double scalar = 1/(a*d - b*c);

    B[0] = scalar * d;
    B[1] = scalar * (-b);
    B[2] = scalar * (-c);
    B[3] = scalar * a;
}

__device__ double pdf(double *x, double *mean, double* cov)
{
    double cov_inv[] = {0, 0, 0, 0};
    pinv(cov, cov_inv);

    double scalar = 1/(2*M_PI*sqrt(cov[0]*cov[3] - cov[1]*cov[2]));

    double m = x[0] - mean[0];
    double n = x[1] - mean[1];

    double arg = m*m*(cov_inv[0]) + n*n*(cov_inv[3]) + m*n*(cov_inv[1] + cov_inv[2]);

    return scalar * exp(-0.5 * arg);
}

__device__ void add_measurement_as_landmark(double *particle, double *measurement, double *measurement_cov)
{
    double pos[] = { particle[0], particle[1] };
    double landmark[] = {0, 0};
    to_coords(particle, measurement, landmark);

    double q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);

    double H[] = {
        (landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)),
        -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
    };

    pinv(H, H);

    double H_inv_t[] = {
        H[0], H[2],
        H[1], H[3]
    };

    double S[] = {
        0, 0, 0, 0
    };

    matmul(H, measurement_cov, S);
    matmul(S, H_inv_t, S);
    add_landmark(particle, landmark, S, measurement[2]);
}


__device__ void add_measurements_as_landmarks(double *particle, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    double *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
    }
}


__device__ void update_landmarks(int id, double *particle, landmark_measurements *measurements, int *in_range, int *n_matches, double range, double fov, double thresh)
{
    double *measurement_cov = measurements->measurement_cov;
    int n_measurements = measurements->n_measurements;

    int n_landmarks = get_n_landmarks(particle);

    for(int i = 0; i < n_measurements; i++) {
        double landmark_idx = -1;

        for(int j = 0; j < n_landmarks; j++) {
            double *p = get_landmark_id(particle, j);
            double id = p[0];
            if(measurements->measurements[i][2] == id) {
                landmark_idx = j;
                break;
            }
        }

        if(landmark_idx != -1) {
            double *landmark = get_mean(particle, landmark_idx);
            double pos[] = { particle[0], particle[1] };
            double theta = particle[2];

            double q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);
            double measurement_predicted[] = {
                sqrt(q), mod_angle(atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta)
            };

            double residual[2] = {
                measurements->measurements[i][0] - measurement_predicted[0],
                measurements->measurements[i][1] - measurement_predicted[1]
            };


            double H[] = {
                (landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)),
                -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
            };

            double Ht[] = {
                H[0], H[2],
                H[1], H[3]
            };

            double S[] = {
                0, 0, 0, 0
            };

            double *landmark_cov = get_cov(particle, landmark_idx);
        
            matmul_jacobian(H, landmark_cov, measurement_cov, S);
            double S_inv[] = {0, 0, 0, 0};
            pinv(S, S_inv);


            double Q[] = {0, 0, 0, 0};
            double K[] = { 0, 0, 0, 0 };
            matmul(landmark_cov, Ht, Q);
            matmul(Q, S_inv, K);

            double K_residual[] = { 0, 0 };
            vecmul(K, residual, K_residual);
            landmark[0] += K_residual[0];
            landmark[1] += K_residual[1];

            double KH[] = { 0, 0, 0, 0};
            matmul(K, H, KH);
            double new_cov[] = { 1 - KH[0], -KH[1], -KH[2], 1 - KH[3] };
            matmul(new_cov, landmark_cov, new_cov);
            landmark_cov[0] = new_cov[0];
            landmark_cov[1] = new_cov[1];
            landmark_cov[2] = new_cov[2];
            landmark_cov[3] = new_cov[3];

            particle[3] *= pdf(measurements->measurements[i], measurement_predicted, S);
            // particle[3] *= 2.0;
            // particle[3] += 1e-38;

        } else {
            add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
        }
    }


}

__global__ void update(
    double *particles, int block_size, int *scratchpad_mem, int scratchpad_size, double measurements_array[][3], int n_particles, int n_measurements,
    double *measurement_cov, double threshold, double range, double fov, int max_landmarks)
{

    if(n_measurements == 0) {
        return;
    }

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int *scratchpad = scratchpad_mem + (2 * thread_id * max_landmarks);
    int *in_range = scratchpad;
    int *n_matches = in_range + max_landmarks;

    landmark_measurements measurements;
    measurements.n_measurements = n_measurements;
    measurements.measurement_cov = measurement_cov;
    measurements.measurements = measurements_array;

    for(int k = 0; k < block_size; k++) {
        int particle_id = thread_id*block_size + k;
        if(particle_id >= n_particles) {
            return;
        }
        
        double *particle = get_particle(particles, particle_id);
        int n_landmarks = get_n_landmarks(particle);
    
        if(n_landmarks == 0) {
            add_measurements_as_landmarks(particle, &measurements);
            continue;
        }

        update_landmarks(particle_id, particle, &measurements, in_range, n_matches, range, fov, threshold);
    }
}