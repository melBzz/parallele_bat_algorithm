#include <math.h>
#include <stdint.h>

#include "bat_rng.h"

/*
 * bat_rng.c
 *
 * Goal : give each Bat its own small random number generator.
 *
 * Why:
 * - The standard C function rand() uses shared global state.
 * - With OpenMP, multiple threads calling rand() can corrupt that state or
 *   produce unpredictable sequences.
 * - Unpredictable randomness changes the algorithm path and can also change how
 *   much work is done (more/less local search), which makes benchmarks unfair.
 *
 * Our solution:
 * - Store one RNG state (a 32-bit integer) inside each Bat.
 * - Update that state whenever we need a random number.
 * - This makes runs reproducible, thread-safe, and comparable across
 *   sequential / OpenMP / MPI.
 *
 * Important: this RNG is for simulation/benchmarking ONLY.
 * It is NOT cryptographically secure.
 */

/*
 * Mixes a 32-bit value to produce a well-scrambled result.
 * Used only to initialize the RNG state, not to generate random sequences.
 *
 * Parameters:
 *   - x : input value to mix
 */
static uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

/*
 * Advances the RNG state using a xorshift transition and returns a 32-bit
 * pseudo-random value. This function is used as the main random generator
 * during the algorithm, after the state has been initialized.
 *
 * Parameters:
 *   - state : pointer to the RNG state to update
 */
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/*
 * Initializes a per-bat RNG state using a global seed and a stream identifier.
 * Ensures a non-zero initial state.
 *
 * Parameters:
 *   - seed      : global random seed
 *   - stream_id : unique identifier for the random stream
 */
uint32_t bat_rng_init(uint32_t seed, uint32_t stream_id) {
    uint32_t s = splitmix32(seed ^ (stream_id * 0xA511E9B3u));
    if (s == 0) {
        s = 0x6D2B79F5u;
    }
    return s;
}

 /*
 * Generates a uniform random value strictly between 0 and 1.
 * Advances the RNG state using the xorshift generator.
 *
 * Parameters:
 *   - state : pointer to the RNG state to update
 */
double bat_rng_uniform01(uint32_t *state) {
    uint32_t r = xorshift32(state);
    return ((double)r + 1.0) / ((double)UINT32_MAX + 2.0);
}

/*
 * Generates a uniform random value in the interval (a, b).
 * Uses a uniform draw in (0,1) and maps it to the given range.
 *
 * Parameters:
 *   - state : pointer to the RNG state to update
 *   - a     : lower bound of the interval
 *   - b     : upper bound of the interval
 */

double bat_rng_uniform(uint32_t *state, double a, double b) {
    return a + (b - a) * bat_rng_uniform01(state);
}

double bat_rng_normal(uint32_t *state, double mean, double stddev) {
    /*
     * Gaussian random number using Box-Muller:
     * - turn two uniform random numbers into one normal random number.
     */
    double u1 = bat_rng_uniform01(state);
    double u2 = bat_rng_uniform01(state);

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z0;
}
