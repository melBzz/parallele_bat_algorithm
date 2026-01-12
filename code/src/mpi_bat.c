#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#include "bat.h"
#include "bat_utils.h"

/*
 * MPI version of the Bat Algorithm.
 *
 * Idea:
 * - We split the bats between MPI processes (each rank has a local part).
 * - Every iteration, each rank updates its local bats using the current global best.
 * - Then each rank finds its local best.
 * - We use MPI_Allreduce with MPI_MAXLOC to find which rank has the best f_value.
 * - Finally, that best bat is broadcast so all ranks use the same global_best.
 *
 * This is the "AllReduce method": the global best score (value + owner rank)
 * is computed collectively with Allreduce.
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

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_bats, max_iters;
    int quiet;
    unsigned int seed;
    parse_args(argc, argv, &n_bats, &max_iters, &seed, &quiet);

    if (n_bats <= 0 || max_iters <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Invalid parameters: n_bats=%d iters=%d\n", n_bats, max_iters);
        }
        MPI_Finalize();
        return 1;
    }

    /* Simple assumption: we want an equal number of bats per process */
    if (n_bats % size != 0) {
        if (rank == 0) {
            printf("N_BATS must be divisible by number of processes\n");
            printf("Hint: choose --n-bats divisible by procs (n_bats=%d, procs=%d)\n", n_bats, size);
        }
        MPI_Finalize();
        return 0;
    }

    int local_n = n_bats / size;

    Bat *all_bats = NULL;
    Bat local_bats[local_n];

    Bat local_best, global_best;

    /* Different seed per rank (so ranks do not generate the same random numbers) */
    srand(seed + (unsigned int)rank * 100U);

    /* ---------- Initialization ---------- */
    if (rank == 0) {
        /* Rank 0 creates and initializes the full population */
        all_bats = malloc((size_t)n_bats * sizeof(Bat));
        initialize_bats(all_bats, n_bats, &global_best);
    }

    /* Split the population: each rank receives local_n bats in local_bats[] */
    MPI_Scatter(
        all_bats,
        local_n * sizeof(Bat),
        MPI_BYTE,
        local_bats,
        local_n * sizeof(Bat),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    /* Synchronize before timing to measure only the parallel region */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* ---------- Main loop ---------- */
    for (int t = 0; t < max_iters; t++) {

        /* 1) Update the bats owned by this rank (local work) */
        for (int i = 0; i < local_n; i++) {
            update_bat(local_bats, local_n, &global_best, i, t);
        }

        /* 2) Find the best bat inside this rank (local best) */
        local_best = local_bats[0];
        for (int i = 1; i < local_n; i++) {
            if (local_bats[i].f_value > local_best.f_value) {
                local_best = local_bats[i];
            }
        }

        /* ---------- Global best using Allreduce ---------- */
        /*
         * We cannot directly Allreduce the whole Bat struct.
         * So we first Allreduce only:
         * - the best f_value
         * - and the rank that owns this best value
         * MPI_MAXLOC returns the maximum value, and also where it was found.
         */
        struct {
            double value;
            int rank;
        } local_data, global_data;

        local_data.value = local_best.f_value;
        local_data.rank  = rank;

        MPI_Allreduce(
            &local_data,
            &global_data,
            1,
            MPI_DOUBLE_INT,
            MPI_MAXLOC,
            MPI_COMM_WORLD
        );

        /* 3) Now we know which rank has the global best value.
         * That rank copies its local_best into global_best, then broadcasts it.
         * After MPI_Bcast, every rank has the same global_best.
         */
        if (rank == global_data.rank) {
            global_best = local_best;
        }

        MPI_Bcast(
            &global_best,
            sizeof(Bat),
            MPI_BYTE,
            global_data.rank,
            MPI_COMM_WORLD
        );

        if (!quiet && rank == 0 && t % 1000 == 0) {
            printf("[Iter %d] Global best = %f\n", t, global_best.f_value);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (!quiet) {
            printf("\nFinal best f_value = %f\n", global_best.f_value);
            printf("Final position = (");
            for (int d = 0; d < dimension; d++) {
                printf("%s%f", d == 0 ? "" : ", ", global_best.x_i[d]);
            }
            printf(")\n");
        }
         printf("BENCH version=mpi n_bats=%d iters=%d procs=%d threads=1 time_s=%.6f\n",
             n_bats, max_iters, size, elapsed);
        free(all_bats);
    }

    MPI_Finalize();
    return 0;
}
