#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "bat.h"
#include "bat_utils.h"

#define N_BATS     30
#define MAX_ITERS  10000

/*
 * OpenMP version of the Bat Algorithm.
 *
 * Idea:
 * - We keep a shared array bats[] in memory.
 * - Each iteration, we update all bats in parallel (omp for).
 * - Each thread tracks its best bat (thread_best).
 * - At the end, we merge the thread bests to get the iteration best (iter_best).
 */

int main(void) {

    srand((unsigned int) time(NULL));

    Bat bats[N_BATS];
    Bat best_bat;

    /* Create initial bats and compute the first best bat */
    initialize_bats(bats, &best_bat);

    for (int t = 0; t < MAX_ITERS; t++) {

        Bat iter_best;
        /* iter_best starts as the best solution from the previous iteration */
        iter_best = best_bat;

        /* Parallel region: multiple threads work together */
        #pragma omp parallel
        {
            /* Each thread keeps its own best bat (private variable) */
            Bat thread_best = iter_best;

            /* Split the bats between threads */
            #pragma omp for
            for (int i = 0; i < N_BATS; i++) {
                /* Update one bat using the best solution known at this moment */
                update_bat(bats, &iter_best, i, t);

                /* Track the best bat seen by this thread */
                if (bats[i].f_value > thread_best.f_value) {
                    thread_best = bats[i];
                }
            }

            /* Merge the thread bests into a single iter_best (one thread at a time) */
            #pragma omp critical
            {
                if (thread_best.f_value > iter_best.f_value) {
                    iter_best = thread_best;
                }
            }
        }

        /* Save the best solution for the next iteration */
        best_bat = iter_best;

        if (t % 100 == 0) {
            printf("[Iter %d] Best f_value = %f\n", t, best_bat.f_value);
        }
    }

    printf("\nFinal best f_value = %f\n", best_bat.f_value);
    printf("Final position = (");
    for (int d = 0; d < dimension; d++) {
        printf("%s%f", (d == 0 ? "" : ", "), best_bat.x_i[d]);
    }
    printf(")\n");

    return 0;
}
