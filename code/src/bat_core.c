#include <stdlib.h>
#include <math.h>
#include "bat.h"
#include "bat_utils.h"
#include "bat_rng.h"

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
 * Initializes the bat population.
 * For each bat, an independent random generator is initialized, an initial
 * position and velocity are assigned, the Bat Algorithm parameters
 * (frequency, loudness, pulse rate) are set, and the objective function
 * is evaluated. The best initial bat is then selected.
 *
 * Parameters:
 *   - bats     : array containing the bat population
 *   - n_bats   : number of bats
 *   - best_bat : output parameter for the initial best bat
 *   - seed     : global random seed
 */

void initialize_bats_seeded(Bat bats[], int n_bats, Bat *best_bat, uint32_t seed) {
   
    for (int i = 0; i < n_bats; i++) {

        /* Initialize RNG state for this bat */
        bats[i].rng_state = bat_rng_init(seed, (uint32_t)i);
        uint32_t *rng = &bats[i].rng_state;

        /* Initial position and velocity */
        for (int d = 0; d < dimension; d++) {
            /* Position starts uniform in [Lb, Ub] (here: [-5, 5]). */
            bats[i].x_i[d] = bat_rng_uniform(rng, -5.0, 5.0);
            bats[i].v_i[d] = V0;
        }


        /* Initialize Bat Algorithm parameters */
        bats[i].f_i = F_MIN;    
        bats[i].A_i = A0;
        bats[i].r_i = R0;

        /* Evaluate objective function at initial position */
        bats[i].f_value = objective_function(bats[i].x_i);
    }

    /* Select the best bat in the initial population */
    int best_index = 0;
    for (int i = 1; i < n_bats; i++) {
        if (bats[i].f_value > bats[best_index].f_value) {
            best_index = i;
        }
    }

    *best_bat = bats[best_index];
}

void initialize_bats(Bat bats[], int n_bats, Bat *best_bat) {
    /* Backward-compatible wrapper (used by older code paths). */
    initialize_bats_seeded(bats, n_bats, best_bat, 1u);
}

/*
 * Updates a single bat for one iteration.
 * The bat moves toward the current global best, optionally tests a local
 * candidate around the global best, and accepts the new position only if
 * it improves the bat and passes the loudness condition.
 *
 * Parameters:
 *   - bats     : array containing the bat population
 *   - n_bats   : number of bats
 *   - best_bat : current global best (read-only)
 *   - i        : index of the bat to update
 *   - t        : current iteration index
 */
void update_bat(Bat bats[], int n_bats, const Bat *best_bat, int i, int t) {

    /* RNG state of bat i */
    uint32_t *rng = &bats[i].rng_state;
    
    /* Random frequency in [F_MIN, F_MAX]. */
    double beta = bat_rng_uniform01(rng);
    bats[i].f_i = F_MIN + (F_MAX - F_MIN) * beta;

    /* Velocity update: move toward global best. */
    for (int d = 0; d < dimension; d++) {
        bats[i].v_i[d] += (best_bat->x_i[d] - bats[i].x_i[d] ) * bats[i].f_i;
    }

    /* Position update + bounds clamp. */
    for (int d = 0; d < dimension; d++) {
        bats[i].x_i[d] += bats[i].v_i[d];
        if (bats[i].x_i[d] < Lb) bats[i].x_i[d] = Lb;
        if (bats[i].x_i[d] > Ub) bats[i].x_i[d] = Ub;
    }

    /* Candidate = position after the global move. */
    double candidate_x[dimension];
    for (int d = 0; d < dimension; d++) {
        candidate_x[d] = bats[i].x_i[d];
    }

    /* Evaluate the candidate obtained from the global move. */
    double Fnew = objective_function(candidate_x);

    /* Optional local search (triggered by pulse rate). */
    double rand_pulse = bat_rng_uniform01(rng);
    if (rand_pulse > bats[i].r_i) {

        double local_x[dimension];
        double A_mean = compute_A_mean(bats, n_bats);

        // local random walk around global best
        for (int d = 0; d < dimension; d++) {
            double eps = bat_rng_normal(rng, 0.0, 1.0);
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

    /* Accept only if improved AND passes loudness test. */
    double rand_loud = bat_rng_uniform01(rng);
    if ((Fnew > bats[i].f_value) && (rand_loud < bats[i].A_i)) {
       
        for (int d = 0; d < dimension; d++) {
            bats[i].x_i[d] = candidate_x[d];
        }
        bats[i].f_value = Fnew;

        /* Update loudness (A_i) and pulse rate (r_i) using alpha, gamma (Yang) */
        bats[i].A_i *= ALPHA;                       // A_i^{t+1} = alpha * A_i^t
        bats[i].r_i = R0 * (1.0 - exp(-GAMMA * t)); // r_i^{t+1} = r0 * (1 - e^{-gamma t})

        /* Caller recomputes the global best outside this function. */
    }
}
