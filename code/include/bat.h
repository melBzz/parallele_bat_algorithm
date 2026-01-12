#ifndef BAT_H
#define BAT_H

#define dimension 2

typedef struct {
    double x_i[dimension];
    double v_i[dimension];
    double f_i;
    double A_i;
    double r_i;
    double f_value;
} Bat;

/* Core Bat Algorithm functions (implemented in src/sequential.c).
 * We declare them here so every version (sequential / OpenMP / MPI) can call them.
 */
void initialize_bats(Bat bats[], Bat *best_bat);
void update_bat(Bat bats[], Bat *best_bat, int i, int t);

#endif
