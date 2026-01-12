#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#include "bat.h"
#include "bat_utils.h"

/*
 * OpenMP version of the Bat Algorithm.
 *
 * Idea:
 * - We keep a shared array bats[] in memory.
 * - Each iteration, we update all bats in parallel (omp for).
 * - Each thread tracks its best bat (thread_best).
 * - At the end, we merge the thread bests to get the iteration best (iter_best).
 */

static void parse_args(int argc, char **argv, int *n_bats, int *max_iters, unsigned int *seed, int *quiet) {
    *n_bats = N_BATS;
    *max_iters = MAX_ITERS;
    *seed = (unsigned int)time(NULL);
    *quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-bats") == 0 && i + 1 < argc) {
            *n_bats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            *max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            *seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--quiet") == 0) {
            *quiet = 1;
        }
    }
}

int main(int argc, char **argv) {

    int n_bats, max_iters;
    int quiet;
    unsigned int seed;
    parse_args(argc, argv, &n_bats, &max_iters, &seed, &quiet);

    if (n_bats <= 0 || max_iters <= 0) {
        fprintf(stderr, "Invalid parameters: n_bats=%d iters=%d\n", n_bats, max_iters);
        return 1;
    }

    srand(seed);

    Bat *bats = malloc((size_t)n_bats * sizeof(Bat));
    if (!bats) {
        perror("malloc bats");
        return 1;
    }
    Bat best_bat;

    /* Create initial bats and compute the first best bat */
    initialize_bats(bats, n_bats, &best_bat);

    double t0 = omp_get_wtime();

    for (int t = 0; t < max_iters; t++) {

        /* iter_best is the best solution from the previous iteration (read-only guide) */
        Bat iter_best = best_bat;
        Bat next_best = iter_best;

        /* Parallel region: multiple threads work together */
        #pragma omp parallel
        {
            /* Each thread keeps its own best bat (private variable) */
            Bat thread_best = iter_best;

            /* Split the bats between threads */
            #pragma omp for
            for (int i = 0; i < n_bats; i++) {
                /* Update one bat using the best solution known at this moment */
                update_bat(bats, n_bats, &iter_best, i, t);

                /* Track the best bat seen by this thread */
                if (bats[i].f_value > thread_best.f_value) {
                    thread_best = bats[i];
                }
            }

            /* Merge the thread bests into a single iter_best (one thread at a time) */
            #pragma omp critical
            {
                if (thread_best.f_value > next_best.f_value) {
                    next_best = thread_best;
                }
            }
        }

        /* Save the best solution for the next iteration */
        best_bat = next_best;

        if (!quiet && t % 100 == 0) {
            printf("[Iter %d] Best f_value = %f\n", t, best_bat.f_value);
        }
    }

    if (!quiet) {
        printf("\nFinal best f_value = %f\n", best_bat.f_value);
        printf("Final position = (");
        for (int d = 0; d < dimension; d++) {
            printf("%s%f", (d == 0 ? "" : ", "), best_bat.x_i[d]);
        }
        printf(")\n");
    }

    double elapsed = omp_get_wtime() - t0;
    int threads = omp_get_max_threads();
    printf("BENCH version=openmp n_bats=%d iters=%d procs=1 threads=%d time_s=%.6f\n",
           n_bats, max_iters, threads, elapsed);

    free(bats);

    return 0;
}
