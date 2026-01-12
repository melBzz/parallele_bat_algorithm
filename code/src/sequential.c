#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "bat.h"
#include "bat_utils.h"

// ------- FONCTION SNAPSHOT ------------------------------------
static void save_snapshot(const char *filename, Bat bats[], int n_bats) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen snapshot");
        return;
    }

    for (int i = 0; i < n_bats; i++) {
        for (int d = 0; d < dimension; d++) {
            fprintf(fp, (d == 0) ? "%f" : ",%f", bats[i].x_i[d]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// -----------------------------------------------------------------

static double seconds_since(const struct timespec *start, const struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) + 1e-9 * (double)(end->tv_nsec - start->tv_nsec);
}

static void parse_args(int argc, char **argv, int *n_bats, int *max_iters, unsigned int *seed, int *do_snapshot, int *quiet) {
    *n_bats = N_BATS;
    *max_iters = MAX_ITERS;
    *seed = (unsigned int)time(NULL);
    *do_snapshot = 1;
    *quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-bats") == 0 && i + 1 < argc) {
            *n_bats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            *max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            *seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--no-snapshot") == 0) {
            *do_snapshot = 0;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            *quiet = 1;
        }
    }
}

int main(int argc, char **argv) {
    int n_bats, max_iters, do_snapshot;
    int quiet;
    unsigned int seed;
    parse_args(argc, argv, &n_bats, &max_iters, &seed, &do_snapshot, &quiet);

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
    initialize_bats(bats, n_bats, &best_bat);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int t = 0; t < max_iters; t++) {

        /* Use the best solution from the previous iteration as a read-only guide */
        Bat best_snapshot = best_bat;
        for (int i = 0; i < n_bats; i++) {
            update_bat(bats, n_bats, &best_snapshot, i, t);
        }

        /* Recompute best after all bats have been updated */
        best_bat = bats[0];
        for (int i = 1; i < n_bats; i++) {
            if (bats[i].f_value > best_bat.f_value) {
                best_bat = bats[i];
            }
        }

        /* snapshots aux itérations choisies */
        if (do_snapshot) {
            if (t == 0) {
                save_snapshot("snapshot_t000.csv", bats, n_bats);
            } else if (t == 2500) {
                save_snapshot("snapshot_t250.csv", bats, n_bats);
            } else if (t == 5000) {
                save_snapshot("snapshot_t500.csv", bats, n_bats);
            } else if (t == 7500) {
                save_snapshot("snapshot_t750.csv", bats, n_bats);
            }
        }

        // afficher toutes les 100 itérations
        if (!quiet && t % 100 == 0) {
            printf("[Iteration %d] Best f_value = %f  Position = (", t, best_bat.f_value);
            for (int d = 0; d < dimension; d++) {
                printf("%s%f", (d == 0 ? "" : ", "), best_bat.x_i[d]);
            }
            printf(")\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = seconds_since(&t0, &t1);

    if (!quiet) {
        printf("Final best f_value = %f\n", best_bat.f_value);
        printf("Final position = (");
        for (int d = 0; d < dimension; d++) {
            printf("%s%f", (d == 0 ? "" : ", "), best_bat.x_i[d]);
        }
        printf(")\n");
    }

    /* Machine-readable benchmark line */
    printf("BENCH version=sequential n_bats=%d iters=%d procs=1 threads=1 time_s=%.6f\n",
           n_bats, max_iters, elapsed);

    free(bats);
    return 0;
}
