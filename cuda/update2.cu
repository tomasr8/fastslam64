#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <curand_kernel.h>

#define M_PI 3.14159265359
#define MIN(a,b) (((a)<(b))?(a):(b))


extern "C" {

__device__ curandState_t* states[<<N_PARTICLES>>];


// This function is only called once to initialize the rngs.
__global__ void init_rng(int seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, i, 0, s);
    states[i] = s;
}


typedef struct 
{
    double (*measurements)[2];
    int n_measurements;
    double *measurement_cov;
} landmark_measurements;

__device__ double mod_angle(double angle) {
    return atan2(sin(angle), cos(angle));
}

__device__ double vecnorm(double *v) {
    return sqrt(v[0]*v[0] + v[1]*v[1]);
}

__device__ bool in_sensor_range(double *position, double *landmark, double range, double fov) {
    double x = position[0];
    double y = position[1];
    double theta = position[2];
    double lx = landmark[0];
    double ly = landmark[1];

    double va[] = {lx - x, ly - y};
    double vb[] = {range * cos(theta), range * sin(theta)};

    if(vecnorm(va) > range) {
        return false;
    }

    double angle = acos(
        (va[0]*vb[0] + va[1]*vb[1])/(vecnorm(va)*vecnorm(vb))
    );

    if(angle <= (fov/2)) {
        return true;
    } else {
        return false;
    }
}

__device__ bool in_large_sensor_range(double *position, double *landmark, double range) {
    double x = position[0];
    double y = position[1];
    double lx = landmark[0];
    double ly = landmark[1];

    double dist_sq = (lx-x)*(lx-x) + (ly-y)*(ly-y);

    return dist_sq < range*range;
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

__device__ double* get_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 6*max_landmarks + i);
}

__device__ void increment_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    double *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] += 1.0;
}

__device__ void decrement_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    double *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] -= 1.0;
}

__device__ int get_n_landmarks(double *particle)
{
    return (int)particle[5];
}

__device__ void add_landmark(double *particle, double mean[2], double *cov)
{
    int n_landmarks = (int)particle[5];
    particle[5] = (double)(n_landmarks + 1);

    double *new_mean = get_mean(particle, n_landmarks);
    double *new_cov = get_cov(particle, n_landmarks);
    double *new_prob = get_landmark_prob(particle, n_landmarks);

    new_mean[0] = mean[0];
    new_mean[1] = mean[1];

    new_cov[0] = cov[0];
    new_cov[1] = cov[1];
    new_cov[2] = cov[2];
    new_cov[3] = cov[3];

    new_prob[0] = 1.0;
}

__device__ void remove_landmark(double *particle, int i)
{
    int n_landmarks = (int)particle[5];

    double *mean_a = get_mean(particle, i);
    double *mean_b = get_mean(particle, n_landmarks - 1);

    mean_a[0] = mean_b[0];
    mean_a[1] = mean_b[1];

    double *cov_a = get_cov(particle, i);
    double *cov_b = get_cov(particle, n_landmarks - 1);

    cov_a[0] = cov_b[0];
    cov_a[1] = cov_b[1];
    cov_a[2] = cov_b[2];
    cov_a[3] = cov_b[3];

    double *prob_a = get_landmark_prob(particle, i);
    double *prob_b = get_landmark_prob(particle, n_landmarks - 1);

    prob_a[0] = prob_b[0];
    
    particle[5] = (double)(n_landmarks - 1);
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

__device__ void inv3(double A[][3], double B[][3])
{
    double a = A[0][0];
    double b = A[0][1];
    double c = A[0][2];
    double d = A[1][0];
    double e = A[1][1];
    double f = A[1][2];
    double g = A[2][0];
    double h = A[2][1];
    double i = A[2][2];
    
    double det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);

    B[0][0] = (e*i - f*h)/det;
    B[0][1] = -(b*i - c*h)/det;
    B[0][2] = (b*f - c*e)/det;
    B[1][0] = -(d*i - f*g)/det;
    B[1][1] = (a*i - c*g)/det;
    B[1][2] = -(a*f - c*d)/det;
    B[2][0] = (d*h - e*g)/det;
    B[2][1] = -(a*h - b*g)/det;
    B[2][2] = (a*e - b*d)/det;
}

__device__ void matmul_mn(int m, int n, int p, double *A, double *B, double *C)
{
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            for(int k = 0; k < n; k++) {
                double prod = (*((A + i*n) + k)) * (*((B + k*p) + j));
                *((C +i*p) + j) += prod;
                // C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

__device__ void get_new_state(
    double *state_mean, double state_sigma[][3], double *S_inv_1d, double Hx[][3], double Hxt[][2], double *measurement, double*measurement_predicted,
    double *new_state_mean, double new_state_sigma[][3]
)
{
    double S_inv[2][2] = {
        {S_inv_1d[0], S_inv_1d[1]},
        {S_inv_1d[2], S_inv_1d[3]}
    };

    // printf("Hxt\n");
    // printf("[%lf, %lf]\n", Hxt[0][0], Hxt[0][1]);
    // printf("[%lf, %lf]\n", Hxt[1][0], Hxt[1][1]);
    // printf("[%lf, %lf]\n", Hxt[2][0], Hxt[2][1]);


    // printf("S_inv\n");
    // printf("[%lf, %lf]\n", S_inv[0][0], S_inv[0][1]);
    // printf("[%lf, %lf]\n", S_inv[1][0], S_inv[1][1]);

    // printf("Hx\n");
    // printf("[%lf, %lf, %lf]\n", Hx[0][0], Hx[0][1], Hx[0][2]);
    // printf("[%lf, %lf, %lf]\n", Hx[1][0], Hx[1][1], Hx[1][2]);

    double HxtSinv[3][2] = {
        {0, 0},
        {0, 0},
        {0, 0}
    };

    
    matmul_mn(3, 2, 2, (double*)Hxt, (double*)S_inv, (double*)HxtSinv);
    // printf("HxtSinv\n");
    // printf("[%lf, %lf]\n", HxtSinv[0][0], HxtSinv[0][1]);
    // printf("[%lf, %lf]\n", HxtSinv[1][0], HxtSinv[1][1]);
    // printf("[%lf, %lf]\n", HxtSinv[2][0], HxtSinv[2][1]);

    double HxtSinvHx[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    matmul_mn(3, 2, 3, (double*)HxtSinv, (double*)Hx, (double*)HxtSinvHx);

    // printf("HxtSinvHx\n");
    // printf("[%lf, %lf, %lf]\n", HxtSinvHx[0][0], HxtSinvHx[0][1], HxtSinvHx[0][2]);
    // printf("[%lf, %lf, %lf]\n", HxtSinvHx[1][0], HxtSinvHx[1][1], HxtSinvHx[1][2]);
    // printf("[%lf, %lf, %lf]\n", HxtSinvHx[2][0], HxtSinvHx[2][1], HxtSinvHx[2][2]);

    double state_sigma_inv[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    inv3(state_sigma, state_sigma_inv);

    // printf("state_sigma_inv\n");
    // printf("[%lf, %lf, %lf]\n", state_sigma_inv[0][0], state_sigma_inv[0][1], state_sigma_inv[0][2]);
    // printf("[%lf, %lf, %lf]\n", state_sigma_inv[1][0], state_sigma_inv[1][1], state_sigma_inv[1][2]);
    // printf("[%lf, %lf, %lf]\n", state_sigma_inv[2][0], state_sigma_inv[2][1], state_sigma_inv[2][2]);

    state_sigma_inv[0][0] += HxtSinvHx[0][0];
    state_sigma_inv[0][1] += HxtSinvHx[0][1];
    state_sigma_inv[0][2] += HxtSinvHx[0][2];
    state_sigma_inv[1][0] += HxtSinvHx[1][0];
    state_sigma_inv[1][1] += HxtSinvHx[1][1];
    state_sigma_inv[1][2] += HxtSinvHx[1][2];
    state_sigma_inv[2][0] += HxtSinvHx[2][0];
    state_sigma_inv[2][1] += HxtSinvHx[2][1];
    state_sigma_inv[2][2] += HxtSinvHx[2][2];

    // double new_state_sigma[][] = {
    //     {0, 0, 0},
    //     {0, 0, 0},
    //     {0, 0, 0}
    // }

    inv3(state_sigma_inv, new_state_sigma);

    // printf("new_state_sigma\n");
    // printf("[%lf, %lf, %lf]\n", new_state_sigma[0][0], new_state_sigma[0][1], new_state_sigma[0][2]);
    // printf("[%lf, %lf, %lf]\n", new_state_sigma[1][0], new_state_sigma[1][1], new_state_sigma[1][2]);
    // printf("[%lf, %lf, %lf]\n", new_state_sigma[2][0], new_state_sigma[2][1], new_state_sigma[2][2]);

    // printf("HxtSinv\n");
    // printf("[%lf, %lf]\n", HxtSinv[0][0], HxtSinv[0][1]);
    // printf("[%lf, %lf]\n", HxtSinv[1][0], HxtSinv[1][1]);
    // printf("[%lf, %lf]\n", HxtSinv[2][0], HxtSinv[2][1]);

    double new_sigma_HxtSinv[3][2] = {
        {0, 0},
        {0, 0},
        {0, 0}
    };

    matmul_mn(3, 3, 2, (double*)new_state_sigma, (double*)HxtSinv, (double*)new_sigma_HxtSinv);
    double dz[2] = { measurement[0] - measurement_predicted[0], mod_angle(measurement[1] - measurement_predicted[1]) };

    // printf("dz [%lf, %lf]\n", dz[0], dz[1]);
    // printf("new_sigma_HxtSinv\n");
    // printf("[%lf, %lf]\n", new_sigma_HxtSinv[0][0], new_sigma_HxtSinv[0][1]);
    // printf("[%lf, %lf]\n", new_sigma_HxtSinv[1][0], new_sigma_HxtSinv[1][1]);
    // printf("[%lf, %lf]\n", new_sigma_HxtSinv[2][0], new_sigma_HxtSinv[2][1]);


    // double new_state_mean[] = { state_mean[0], state_mean[1], state_mean[2] };
    // printf("diff: [%lf, %lf, %lf]\n",
    //     (new_sigma_HxtSinv[0][0]*dz[0] + new_sigma_HxtSinv[0][1]*dz[1]),
    //     (new_sigma_HxtSinv[1][0]*dz[0] + new_sigma_HxtSinv[1][1]*dz[1]),
    //     (new_sigma_HxtSinv[2][0]*dz[0] + new_sigma_HxtSinv[2][1]*dz[1])
    // );

    new_state_mean[0] = state_mean[0] + (new_sigma_HxtSinv[0][0]*dz[0] + new_sigma_HxtSinv[0][1]*dz[1]);
    new_state_mean[1] = state_mean[1] + (new_sigma_HxtSinv[1][0]*dz[0] + new_sigma_HxtSinv[1][1]*dz[1]);
    new_state_mean[2] = state_mean[2] + (new_sigma_HxtSinv[2][0]*dz[0] + new_sigma_HxtSinv[2][1]*dz[1]);
    new_state_mean[2] = mod_angle(new_state_mean[2]);
}

__device__ void add_motion_fix(double Hx[][3], double Hxt[][2], double Q[][3], double HxQHxt[][2])  {
    double HxQ[2][3] = {
        {0, 0, 0},
        {0, 0, 0}
    };

    matmul_mn(2, 3, 3, (double*)Hx, (double*)Q, (double*)HxQ);
    matmul_mn(2, 3, 2, (double*)HxQ, (double*)Hxt, (double*)HxQHxt);
}

__device__ double pdf(double *x, double *mean, double* cov)
{
    double cov_inv[] = {0, 0, 0, 0};
    pinv(cov, cov_inv);

    double scalar = 1/(2*M_PI*sqrt(cov[0]*cov[3] - cov[1]*cov[2]));

    double m = x[0] - mean[0];
    double n = mod_angle(x[1] - mean[1]);

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

    add_landmark(particle, landmark, S);
}


__device__ void add_measurements_as_landmarks(double *particle, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    double *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
    }
}


__device__ double compute_dist(double *particle, int i, double *measurement, double *measurement_cov)
{
    double *landmark = get_mean(particle, i);
    double measurement_xy[] = {0, 0};
    to_coords(particle, measurement, measurement_xy);

    double dist = sqrt(
        (landmark[0] - measurement_xy[0])*(landmark[0] - measurement_xy[0]) +
        (landmark[1] - measurement_xy[1])*(landmark[1] - measurement_xy[1])
    );
    
    // printf("[%f, %f], [%f, %f] -> %f\n", landmark[0], landmark[1], measurement_xy[0], measurement_xy[1], dist);

    return dist;
}

__device__ void predict_from_model(double *particle, double ua, double ub, double dt) {
    if(ua == 0.0 && ub == 0.0) {
        return;
    }

    double angle = particle[2];

    particle[2] += (ua * dt);
    // particle[2] = fmod(particle[2], (double)(2*M_PI));
    particle[2] = atan2(sin(particle[2]), cos(particle[2]));

    double dist = (ub * dt);
    particle[0] += cos(angle) * dist;
    particle[1] += sin(angle) * dist;
}


__device__ void update_landmarks(int id, double *particle, landmark_measurements *measurements, int *in_range, int *n_matches, double range, double fov, double thresh)
{
    double *measurement_cov = measurements->measurement_cov;
    int n_measurements = measurements->n_measurements;

    int n_landmarks = get_n_landmarks(particle);

    int n_in_range = 0;
    for(int i = 0; i < n_landmarks; i++) {
        n_matches[i] = 0;
        double *mean = get_mean(particle, i);
        in_range[n_in_range] = i;
        n_in_range++;
        // if(in_sensor_range(particle, mean, range, fov)) {
        //     in_range[n_in_range] = i;
        //     n_in_range++;
        // }
    }

    
    double state_mean[] = { particle[0], particle[1], particle[2] };
    double P[3][3] = { {0.001, 0, 0}, {0, 0.001, 0}, {0, 0, 0.0005} };
    double state_sigma[3][3] = { {0.001, 0, 0}, {0, 0.001, 0}, {0, 0, 0.0005} };

    if(id == 0) {
        printf("Initial mean [%lf, %lf, %lf]\n", state_mean[0], state_mean[1], state_mean[2]);
    }

    for(int i = 0; i < n_measurements; i++) {
        double best = 1000000.0;
        int best_idx = -1;

        for(int j = 0; j < n_in_range; j++) {
            double dist = compute_dist(particle, in_range[j], measurements->measurements[i], measurement_cov);
            // printf("dist[%d, %d] = %f\n", j, i, dist);

            if(dist <= thresh && dist < best) {
                best = dist;
                best_idx = in_range[j];
            }
        }


        if(best_idx != -1) {
            n_matches[best_idx]++;
        }


        if(best_idx != -1) {
            double *landmark = get_mean(particle, best_idx);
            double pos[] = { particle[0], particle[1] };
            double theta = particle[2];

            double q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);
            double measurement_predicted[] = {
                sqrt(q), mod_angle(atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta)
            };

            double residual[2] = {
                measurements->measurements[i][0] - measurement_predicted[0],
                mod_angle(measurements->measurements[i][1] - measurement_predicted[1])
            };

            // printf("residual[%f, %f]\n", residual[0], residual[1]);

            double H[] = {
                (landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)),
                -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
            };

            double Ht[] = {
                H[0], H[2],
                H[1], H[3]
            };

            double Hx[2][3] = {
                {(landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)), 0},
                {-(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q, -1}
            };

            double Hxt[3][2] = {
                {Hx[0][0], Hx[1][0]},
                {Hx[0][1], Hx[1][1]},
                {Hx[0][2], Hx[1][2]}
            };

            double S[] = {
                0, 0, 0, 0
            };

            double *landmark_cov = get_cov(particle, best_idx);
        
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

            double new_state_sigma[3][3] = {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            };

            double new_state_mean[] = {0, 0, 0};

            get_new_state(
                state_mean, state_sigma, S_inv, Hx, Hxt, measurements->measurements[i], measurement_predicted,
                new_state_mean, new_state_sigma
            );

            // printf("Old state: [%lf %lf %lf]\n", state_mean[0], state_mean[1], state_mean[2]);
            state_mean[0] = new_state_mean[0];
            state_mean[1] = new_state_mean[1];
            state_mean[2] = new_state_mean[2];
            // printf("New state: [%lf %lf %lf]\n", state_mean[0], state_mean[1], state_mean[2]);
            // printf("===============\n");


            state_sigma[0][0] = new_state_sigma[0][0];
            state_sigma[0][1] = new_state_sigma[0][1];
            state_sigma[0][2] = new_state_sigma[0][2];
            state_sigma[1][0] = new_state_sigma[1][0];
            state_sigma[1][1] = new_state_sigma[1][1];
            state_sigma[1][2] = new_state_sigma[1][2];
            state_sigma[2][0] = new_state_sigma[2][0];
            state_sigma[2][1] = new_state_sigma[2][1];
            state_sigma[2][2] = new_state_sigma[2][2];

            particle[0] = state_mean[0];
            particle[1] = state_mean[1];
            particle[2] = state_mean[2];

            if(id == 0) {
                printf("Temp mean [%lf, %lf, %lf]\n", state_mean[0], state_mean[1], state_mean[2]);
                printf("=========\n");
            }

            double HxSHxt[2][2] = {
                {0, 0},
                {0, 0}
            };
            add_motion_fix(Hx, Hxt, P, HxSHxt);
            S[0] += HxSHxt[0][0];
            S[1] += HxSHxt[0][1];
            S[2] += HxSHxt[1][0];
            S[3] += HxSHxt[1][1];

            particle[3] *= pdf(measurements->measurements[i], measurement_predicted, S);

            increment_landmark_prob(particle, best_idx);

        } else {
            add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
        }
    }

    if(id == 0) {
        printf("Final mean [%lf, %lf, %lf]\n", state_mean[0], state_mean[1], state_mean[2]);
        // printf("[%lf, %lf, %lf]\n", state_sigma[0][0], state_sigma[0][1], state_sigma[0][2]);
        // printf("[%lf, %lf, %lf]\n", state_sigma[1][0], state_sigma[1][1], state_sigma[1][2]);
        // printf("[%lf, %lf, %lf]\n", state_sigma[2][0], state_sigma[2][1], state_sigma[2][2]);
        printf("=========\n");
    }

    particle[0] = state_mean[0] + sqrt(state_sigma[0][0]) * curand_normal(states[id]);
    particle[1] = state_mean[1] + sqrt(state_sigma[1][1]) * curand_normal(states[id]);
    particle[2] = mod_angle(state_mean[2] + sqrt(state_sigma[2][2]) * curand_normal(states[id]));
}

__global__ void update(
    double *particles, int *scratchpad_mem, int scratchpad_size, double measurements_array[][2], int n_particles, int n_measurements,
    double *measurement_cov, double threshold, double range, double fov, int max_landmarks, double ua, double ub, double dt)
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

    int particle_id = thread_id;
    if(particle_id >= n_particles) {
        return;
    }
    
    double *particle = get_particle(particles, particle_id);
    int n_landmarks = get_n_landmarks(particle);

    predict_from_model(particle, ua, ub, dt);

    if(n_landmarks == 0) {
        add_measurements_as_landmarks(particle, &measurements);
        return;
    }

    update_landmarks(particle_id, particle, &measurements, in_range, n_matches, range, fov, threshold);
}

}