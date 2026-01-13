#include <stdlib.h>
#include <math.h>
#include "bat.h"
#include "bat_utils.h"

/*
 * Shared algorithm core used by sequential / OpenMP / MPI front-ends.
 *
 * Important design choice:
 * - update_bat() does NOT update the global best directly.
 *   The caller recomputes/updates the best outside the update loop.
 *   This keeps the core function thread-safe for OpenMP and easier to
 *   reason about for MPI.
 */

/* Helper function for update_bat(): average loudness across the population */
static double compute_A_mean(Bat bats[], int n_bats) {
    double sum = 0.0;
    for (int k = 0; k < n_bats; k++) sum += bats[k].A_i;
    return sum / (double)n_bats;
}

/* 
 * Initialize the bat population and find an initial best bat 
 */
void initialize_bats(Bat bats[], int n_bats, Bat *best_bat) {
    /* Initialize bats with random positions and default parameters. */
    for (int i = 0; i < n_bats; i++) {

        // x_i and v_i
        for (int d = 0; d < dimension; d++) {
            /* Position starts uniform in [Lb, Ub] (here: [-5, 5]). */
            bats[i].x_i[d] = uniform_random(-5.0, 5.0);
            bats[i].v_i[d] = V0;
        }


        // frequency, loudness, pulse rate
        bats[i].f_i = F_MIN;      // will be updated later in the loop
        bats[i].A_i = A0;
        bats[i].r_i = R0;

        // evaluate f(x_i)
        bats[i].f_value = objective_function(bats[i].x_i);
    }

    /* Find initial best bat (we maximize f_value). */
    int best_index = 0;
    for (int i = 1; i < n_bats; i++) {
        if (bats[i].f_value > bats[best_index].f_value) {
            best_index = i;
        }
    }

    *best_bat = bats[best_index];  // copy best bat
}

/* 
 * Update bat logic 
 */
void update_bat(Bat bats[], int n_bats, const Bat *best_bat, int i, int t) {
    
    // 1. Update frequency
    double beta = uniform_random(0.0, 1.0);
    bats[i].f_i = F_MIN + (F_MAX - F_MIN) * beta;

    // 2. Update velocity (towards best solution)
    for (int d = 0; d < dimension; d++) {
        bats[i].v_i[d] += (best_bat->x_i[d] - bats[i].x_i[d] ) * bats[i].f_i;
    }

    // 3. Update position
    for (int d = 0; d < dimension; d++) {
        bats[i].x_i[d] += bats[i].v_i[d];

        // apply bounds
        if (bats[i].x_i[d] < Lb) bats[i].x_i[d] = Lb;
        if (bats[i].x_i[d] > Ub) bats[i].x_i[d] = Ub;
    }


    double candidate_x[dimension];

    // start from current position
    for (int d = 0; d < dimension; d++) {
        candidate_x[d] = bats[i].x_i[d];
    }

    /* Evaluate the candidate obtained from the global move. */
    double Fnew = objective_function(candidate_x);

    double rand_pulse = uniform_random(0.0, 1.0);
    if (rand_pulse > bats[i].r_i) {

        double local_x[dimension];

        double A_mean = compute_A_mean(bats, n_bats);

        // local random walk around global best
        for (int d = 0; d < dimension; d++) {
            double eps = normal_random(0.0, 1.0);       // randn(1,d)
            local_x[d] = best_bat->x_i[d] + 0.1 * eps * A_mean;

            /* Clamp the local candidate to bounds. */
            if (local_x[d] < Lb) local_x[d] = Lb;
            if (local_x[d] > Ub) local_x[d] = Ub;
        }
        /* Evaluate the local (random-walk) candidate. */
        double F_local = objective_function(local_x);

        /* If the local candidate is better, keep it as the new candidate. */
        if (F_local > Fnew) {   /* we maximize */
            for (int d = 0; d < dimension; d++) {
                candidate_x[d] = local_x[d];
            }
            Fnew = F_local;
        }
    }

    // ----- Acceptance by loudness (for this bat) -----
    double rand_loud = uniform_random(0.0, 1.0);

    if ((Fnew > bats[i].f_value) && (rand_loud < bats[i].A_i)) {
        // accept candidate as new position of bat i
        for (int d = 0; d < dimension; d++) {
            bats[i].x_i[d] = candidate_x[d];
        }
        bats[i].f_value = Fnew;

        // update A_i and r_i using alpha, gamma (Yang)
        bats[i].A_i *= ALPHA;                       // A_i^{t+1} = alpha * A_i^t
        bats[i].r_i = R0 * (1.0 - exp(-GAMMA * t)); // r_i^{t+1} = r0 * (1 - e^{-gamma t})

        /* Caller recomputes the global best outside this function. */
    }
}
