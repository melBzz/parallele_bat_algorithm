#include <math.h>
#include <stdint.h>

#include "bat_rng.h"

/*
 * bat_rng.c
 *
 * Goal (simple): give each Bat its own small random number generator.
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
 * SplitMix32: used only to create a good starting state for each bat.
 *
 * Intuition: if you seed bat #0 and bat #1 with very similar numbers,
 * you still want their RNG streams to be very different.
 * SplitMix32 does a few mixing operations to "spread" bits well.
 */
static uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

/*
 * Xorshift32: the core RNG step.
 *
 * What is it?
 * - A very small pseudo-random generator that updates a 32-bit state using
 *   XOR (^) and bit shifts (<<, >>).
 * - It is popular in teaching/examples because it is fast and easy to implement.
 *
 * How it works (high level):
 * - Take the current state.
 * - Mix its bits using a few XOR+shift operations.
 * - The result becomes the new state and also the next random integer.
 *
 * Requirement: state must never be 0, otherwise it stays 0 forever.
 */
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

uint32_t bat_rng_init(uint32_t seed, uint32_t stream_id) {
    /*
     * Mix seed and stream id (bat index), then ensure non-zero state.
     * This gives each bat a different deterministic random stream.
     */
    uint32_t s = splitmix32(seed ^ (stream_id * 0xA511E9B3u));
    if (s == 0) {
        s = 0x6D2B79F5u;
    }
    return s;
}

double bat_rng_uniform01(uint32_t *state) {
    /*
     * Return u in (0,1) (strictly inside the interval).
     *
     * Why not [0,1]?
     * - For the normal distribution (Box-Muller) we compute log(u).
     * - log(0) is not defined, so we must never return 0.
     */
    uint32_t r = xorshift32(state);
    return ((double)r + 1.0) / ((double)UINT32_MAX + 2.0);
}

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
