# Parallel Bat Algorithm

This project implements the **Bat Algorithm** optimization method in three versions:
1.  **Sequential** (C)
2.  **OpenMP** (Shared Memory Parallelization)
3.  **MPI** (Message Passing Interface - Distributed Memory)

The goal is to compare the performance and behavior of these parallel implementations against the sequential baseline.

## 📂 Project Structure

```
code/
├── src/
│   ├── sequential.c    # Main entry for Sequential version
│   ├── openmp_bat.c    # Main entry for OpenMP version
│   ├── mpi_bat.c       # Main entry for MPI version
│   ├── bat_core.c      # Core algorithm logic (shared)
│   ├── bat_utils.c     # Helper functions (objective function, math)
│   └── bat_rng.c       # Deterministic RNG used by the core
├── include/
│   ├── bat.h           # Data structures and constants
│   ├── bat_utils.h     # Function prototypes
│   └── bat_rng.h       # RNG prototypes
├── job.pbs             # PBS script for HPC execution
├── benchmark.pbs       # PBS script for benchmarking
└── Makefile            # Build system
```

## 💻 Local Compilation & Usage

To compile the code locally, you need `gcc` and an MPI implementation (like `mpich` or `openmpi`).

### 1. Compile
Navigate to the `code` directory:
```bash
cd code
```

- **Sequential**:
  ```bash
  make
  ```
- **OpenMP**:
  ```bash
  make openmp
  ```
- **MPI**:
  ```bash
  make mpi
  ```

### 2. Run Locally

- **Sequential**:
  ```bash
  ./sequential
  ```
- **OpenMP**:
  ```bash
  # Run with 4 threads
  export OMP_NUM_THREADS=4
  ./openmp_bat
  ```
- **MPI**:
  ```bash
  # Run with 4 processes
  mpiexec -n 4 ./mpi_bat
  ```

## 📈 Benchmarking (Time, Speedup, Efficiency)

The programs print a machine-readable line at the end of each run:

```
BENCH version=<sequential|openmp|mpi> n_bats=<N> iters=<T> procs=<P> threads=<K> time_s=<seconds>
```

For benchmarking, use `--quiet` to disable iteration printing (printing can distort timings).

Examples:

```bash
./sequential --n-bats 2000 --iters 5000 --seed 1 --quiet --no-snapshot

OMP_NUM_THREADS=4 ./openmp_bat --n-bats 2000 --iters 5000 --seed 1 --quiet

mpiexec -n 4 ./mpi_bat --n-bats 2000 --iters 5000 --seed 1 --quiet
```

---

## 🚀 Execution on UNITN HPC Cluster

The instructions below outline how to run this project on the University of Trento HPC environment.

### 1. Connection Requirements
1.  **VPN**: Ensure you are connected to the university network via the **GlobalProtect** VPN agent.
2.  **SSH**: Connect to the HPC login node using your student credentials.
    ```bash
    ssh name.surname@hpc3-login.unitn
    # Enter your password when prompted
    ```

### 2. Transferring Code
You can transfer your source code to the cluster using `scp` or by using VS Code Remote - SSH.
```bash
# Example from your local machine
scp -r parallel-bat-algorithm name.surname@hpc3-login.unitn:~/
```

### 3. Compilation on HPC
Once logged in, verify you have the necessary modules loaded.

**IMPORTANT: Module Handling**
The specific module names (e.g., `gcc`, `openmpi`) vary by cluster.
1.  Check available modules:
    ```bash
    module avail gcc
    module avail mpi
    ```
2.  Load the specific versions found. Common examples on UNITN clusters:
    ```bash
    module load gcc91      # or similar (like gnu, gcc/9.1.0)
    module load openmpi3   # or similar (like openmpi/4.0.3)
    ```
    *If `module load gcc` fails, try finding the exact name using `module avail`.*

3.  **Compile**:
    ```bash
    cd parallel-bat-algorithm/code
    make clean
    make
    make openmp
    make mpi
    ```

### 4. Submitting a Job (PBS)
Direct execution on the login node is discouraged for heavy computations. Use the provided **PBS script** (`job.pbs`) to submit a job to the compute nodes.

1.  **Edit `job.pbs`** (Optional):
    - Adjust `#PBS -l select=1:ncpus=4:mpiprocs=4` to change resources (CPUs/Processes).
    - Ensure the module loading lines match the cluster's environment.

2.  **Submit the Job**:
    ```bash
    qsub job.pbs
    ```

3.  **Check Status**:
    ```bash
    qstat -u name.surname
    ```

4.  **View Results**:
    When the job finishes, two files will be created in the current directory:
    - `output_bat.txt`: Contains the standard output (results).
    - `error_bat.txt`: Contains errors or logs.

    ```bash
    cat output_bat.txt
    ```

  ### Benchmark PBS Job

  To run a strong-scaling and weak-scaling benchmark campaign on the cluster, use:

  ```bash
  qsub benchmark.pbs
  ```

  It writes results to `bench_out.txt` (including multiple `BENCH ...` lines) and `bench_err.txt` in case of errors.

  ### Plotting

  After downloading `bench_out.txt` to your laptop, generate CSV + graphs with:

  ```bash
  python3 tools/bench_analyze.py --input code/bench_out.txt --outdir bench_out
  ```

  The script produces:
  - `bench_out/bench_metrics.csv` with all computed metrics
  - a small set of *combined* comparison plots (sequential vs OpenMP vs MPI), e.g.:
    - `compare_strong_time_...png`
    - `compare_strong_speedup_vs_seq_...png` and `compare_strong_speedup_vs_self_...png`
    - `compare_weak_time_...png`
    - `compare_weak_efficiency_vs_seq_...png` and `compare_weak_efficiency_vs_self_...png`

  Notes about baselines (important for the report):
  - **vs sequential baseline**: compares MPI/OpenMP to the sequential program.
  - **vs self baseline**: compares MPI to MPI(p=1) and OpenMP to OpenMP(p=1).
    This is often the fairest view because different programs can have different overheads.

  If you do not have matplotlib installed:

  ```bash
  pip install matplotlib # Windows distributions
  sudo apt install python3-matplotlib # Linux distributions (or use a virtual environment)
  ```

## 🎲 About Randomness (and “Xorshift32”)

The Bat Algorithm is **stochastic** (it uses random numbers), so for benchmarking we need the random number generation to be:
- **deterministic** (same seed = same run)
- **thread-safe** (OpenMP should not break it)

Originally, the code used the C function `rand()`. In OpenMP this is problematic because `rand()` is shared global state.
Two threads calling it at the same time can give unpredictable behavior and unreliable timings.

To fix this, we added a small deterministic RNG in `src/bat_rng.c`:
- **Xorshift32** is just a tiny pseudo-random generator based on bit operations (XOR and shifts).
- It is **fast** and simple, and we use it only for experiments/benchmarking (it is not cryptography).

Each `Bat` stores its own RNG state, so each bat generates its own random numbers independently.
This makes sequential/OpenMP/MPI runs comparable and stable.

## 📝 Implementation Details

- **Sequential**: The standard Bat Algorithm loop.
- **OpenMP**: Parallelizes the inner loop over the population of bats. Each thread tracks its own "local best" and updates a shared iteration best inside a critical section.
- **MPI**: Uses `MPI_Scatter` to distribute bats among processes. Uses `MPI_Allreduce` with `MPI_MAXLOC` to find the global best fitness and its owner efficiently.

For fairness and reproducibility, all versions initialize the population using a fixed `--seed` value and the same deterministic per-bat RNG.
