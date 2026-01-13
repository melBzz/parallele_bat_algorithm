#ifndef BAT_RNG_H
#define BAT_RNG_H

#include <stdint.h>

/*
 * bat_rng.h
 *
 * Small deterministic random-number utilities.
 *
 * This project is an optimization algorithm (Bat Algorithm), so it needs
 * randomness. For benchmarking and for OpenMP correctness, we need this
 * randomness to be:
 * - reproducible (same --seed => same run)
 * - thread-safe (OpenMP)
 * - cheap (called many times)
 *
 * We therefore avoid C's rand() and instead store a RNG state per Bat.
 *
 * Note: this is NOT cryptography. It's only meant for simulation/experiments.
 */

/* Initialize a per-bat RNG state from a global seed + an index (e.g., bat id). */
uint32_t bat_rng_init(uint32_t seed, uint32_t stream_id);

/* Uniform random in (0,1) (never returns exactly 0 or 1). */
double bat_rng_uniform01(uint32_t *state);

/* Uniform random in [a,b]. */
double bat_rng_uniform(uint32_t *state, double a, double b);

/* Gaussian random using Box-Muller. */
double bat_rng_normal(uint32_t *state, double mean, double stddev);

#endif
