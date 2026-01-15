#ifndef BAT_H
#define BAT_H

#include <stdint.h>

#define dimension 2

/* Default values (can be overridden at runtime via CLI options). */
#define N_BATS     40
#define MAX_ITERS  10000

#define F_MIN      0.0
#define F_MAX      1.0

#define A0         1.0    // initial loudness
#define R0         1.0    // initial pulse rate
#define V0         0.0

#define ALPHA      0.97
#define GAMMA      0.1

#define Ub         5
#define Lb         -5

typedef struct {
    double x_i[dimension]; // position
    double v_i[dimension]; // velocity
    double f_i;            // frequency
    double A_i;            // loudness
    double r_i;            // pulse rate
    double f_value;        // fitness value

    /* Per-bat RNG state (makes OpenMP/MPI runs deterministic and thread-safe). */
    uint32_t rng_state;
} Bat; // Structure representing a bat in the Bat Algorithm

/* Core Bat Algorithm functions (implemented in src/bat_core.c).
 * These are shared by sequential / OpenMP / MPI implementations.
 */
void initialize_bats(Bat bats[], int n_bats, Bat *best_bat);
void update_bat(Bat bats[], int n_bats, const Bat *best_bat, int i, int t);

/* Deterministic initializer used by all front-ends. */
void initialize_bats_seeded(Bat bats[], int n_bats, Bat *best_bat, uint32_t seed);

#endif
