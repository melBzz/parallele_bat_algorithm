# Introduction

This work is carried out within the framework of the **EIT Digital** program and the course **High‑Performance Computing for Data Science**. The objective of the project is to conduct an in‑depth analysis of an optimization algorithm, to design a **sequential implementation**, and subsequently to develop a **parallelized version** suitable for **HPC environments**.

We chose to study the **Bat Algorithm (BA)**, a bio‑inspired metaheuristic proposed by **Xin‑She Yang (2010)**, whose mechanism is based on a simplified mathematical modeling of the **echolocation behavior of microbats**. The goal of this first phase is to thoroughly understand the internal mechanisms of the algorithm, its parameters, and the way it explores the search space in order to solve **continuous optimization problems**.

Continuous optimization problems arise in many scientific and industrial domains and consist in finding, among all possible configurations, the one that minimizes or maximizes an objective function. As highlighted by **Nocedal and Wright** in *Numerical Optimization*, such problems can be particularly challenging when the objective function is **non‑convex**, **non‑differentiable**, **noisy**, or when the **dimensionality of the problem increases**. In these cases, gradient‑based methods often become ineffective.

A comprehensive review by **Rios and Sahinidis (2013)** further emphasizes that when derivatives are unavailable or unreliable, classical optimization methods are “*of little or no use*”, and that their performance deteriorates rapidly as the problem dimension grows. These limitations motivate the use of **derivative‑free** and **metaheuristic** approaches.

As a stochastic algorithm, the Bat Algorithm incorporates randomness to efficiently explore the search space. Its design explicitly balances **global exploration** and **local exploitation**, making it well‑suited for highly nonlinear objective functions and multimodal landscapes with multiple local optima.

---

# Principles of the Bat Algorithm

The Bat Algorithm is inspired by the echolocation mechanism used by microbats to navigate and hunt in complete darkness. In nature, bats emit ultrasonic pulses and analyze the returning echoes to estimate the distance, direction, size, and motion of nearby objects. As they approach a target, they adapt the **frequency**, **loudness**, and **pulse emission rate** of these signals.

Yang translates these biological observations into algorithmic concepts. Each bat is modeled as a candidate solution characterized by:

* a **position** $x_i$ in the search space,
* a **velocity** $v_i$,
* a **frequency** $f_i$ controlling the scale of movement,
* a **loudness** $A_i$ governing solution acceptance,
* a **pulse rate** $r_i$ controlling the switch between exploration and exploitation.

At each iteration, these parameters are updated stochastically, simulating the collective behavior of a bat colony progressively converging toward an optimal region.

---

# Echolocation Model and Biological Motivation

Microbats emit ultrasonic pulses with frequencies typically ranging from **25 kHz to 150 kHz**, each pulse lasting only a few milliseconds. During the search phase, bats emit around **10–20 pulses per second**, increasing up to **200 pulses per second** as they approach prey. Pulse loudness may reach **110 dB**, but decreases as the bat nears its target to improve precision and avoid sensory saturation.

From the reflected echoes, bats infer:

* **distance**, via time‑of‑flight measurement,
* **direction**, via interaural time differences,
* **target characteristics**, via echo intensity and Doppler effects.

The distance estimation follows the classical acoustic relation:

$$
\text{distance} = v \cdot \frac{t}{2},
$$

where $v$ is the speed of sound and $t$ is the round‑trip travel time of the signal.

This remarkable sensing capability motivates the abstraction used in the Bat Algorithm, where objective‑function evaluations play the role of echo analysis.

---

# Mathematical Formulation of the Bat Algorithm

Following Yang (2010), the Bat Algorithm is based on three idealized assumptions:

1. Each bat can evaluate the quality of its current position through the objective function.
2. Bats adapt their frequency, velocity, and pulse rate depending on their proximity to promising solutions.
3. Loudness decreases while pulse rate increases over time, modeling a transition from exploration to exploitation.

The **global update** equations are:

$$
f_i = f_{\min} + (f_{\max} - f_{\min}) , \beta, \quad \beta \sim U(0,1),
$$

$$
v_i^{t+1} = v_i^t + (x^* - x_i^t) , f_i,
$$

$$
x_i^{t+1} = x_i^t + v_i^{t+1},
$$

where $x^*$ denotes the current best solution.

The **local search** mechanism is defined as:

$$
x_{\text{new}} = x_{\text{best}} + \sigma , \varepsilon_t , A^{(t)},
$$

with $\varepsilon_t \sim \mathcal{N}(0,1)$ and $\sigma$ a scaling parameter.

The adaptive behavior of loudness and pulse rate follows:

$$
A_i^{t+1} = \alpha A_i^t, \quad r_i^{t+1} = r_{0i} \bigl(1 - e^{-\gamma t}\bigr),
$$

ensuring a gradual shift from exploration to exploitation.

---

# Algorithmic Structure

**Algorithm: Bat Algorithm**

**Input:** Objective function $f(x)$
**Output:** Best solution found

1. Initialize a population of $n$ bats with positions $x_i$ and velocities $v_i$.
2. Initialize frequencies $f_i$, loudness $A_i$, and pulse rates $r_i$.
3. While $t < T_{\max}$:

   * Update frequencies, velocities, and positions.
   * With probability $1 - r_i$, generate a local solution around the current best.
   * Evaluate candidate solutions and apply acceptance criteria based on $A_i$.
   * Update loudness and pulse rate if improvement occurs.
   * Update the global best solution $x^*$.
4. Return $x^*$.

---

# Conceptual Limitations and Critical Analysis

A careful examination of Yang’s original formulation reveals several **conceptual inconsistencies**, particularly regarding the interpretation of the **frequency** $f_i$ and the **pulse rate** $r_i$.

* In biological systems, **higher frequencies correspond to fine‑grained localization**, whereas in the algorithm, higher $f_i$ values produce **larger displacements**, which is more characteristic of global exploration.
* The condition `rand > r_i` used to trigger local search appears counter‑intuitive if $r_i$ is interpreted as increasing near the optimum.

Moreover, an inconsistency exists between the velocity update equation in the original paper and the MATLAB implementation provided by Yang, leading to a **sign compensation effect** between frequency and velocity updates. This ambiguity complicates the theoretical interpretation of the algorithm and motivates careful implementation choices.

---

# Sequential Implementation Choices

Our sequential C implementation is inspired by:

* Yang’s original 2010 paper,
* the MATLAB reference implementation from *Nature‑Inspired Optimization Algorithms* (2014),
* and the clarified equations presented in the 2020 edition.

We explicitly correct the velocity update to ensure attraction toward the best solution, while preserving the original pulse‑rate condition. Two candidate solutions (global and local) are evaluated separately at each iteration, improving algorithmic clarity compared to the MATLAB version, which overwrites the global solution when local search is triggered.

Parameter choices (population size, frequency bounds, loudness, pulse rate, domain limits) strictly follow Yang’s recommendations, with minor adaptations justified by the characteristics of the chosen objective function.

---

*To do:*

* Refine and standardize academic citations.
* Insert and document the sequential C implementation.

-----

# Parallel Design

#### Analysis of Parallelism in the Bat Algorithm

The **Bat Algorithm** is a population-based metaheuristic in which a set of individuals (bats) evolve simultaneously within the search space.  
At each iteration, every individual updates its position, velocity, and internal parameters, and then evaluates the objective function associated with its current position.

Analysis of the sequential algorithm shows that most of these operations are **independent for each individual**. In particular, the following steps exhibit no direct dependencies between bats:

- updating the frequency, velocity, and position;
- evaluating the objective function;
- updating the *loudness* and *pulse rate* parameters.

These operations constitute the most computationally expensive part of the algorithm and naturally lend themselves to **data parallelism**, where different individuals can be processed simultaneously on distinct computing units.

However, the algorithm introduces a **global dependency** through the best solution found so far, denoted \( x_{\text{best}} \). This information is used by all individuals to guide local search. Consequently, although the algorithm is largely parallelizable, a **global synchronization** is required to ensure consistency of this information across the different computing units.

#### Master–Worker Model

A first parallelization strategy for the Bat Algorithm is based on a **master–worker model**, as proposed by Noor et al. in *[Performance of Parallel Distributed Bat Algorithm using MPI on a PC Cluster](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3559955)*. From our perspective, this approach appears to be the most natural solution for parallelizing the Bat Algorithm in a distributed MPI environment.

##### General Principle

In this model, the global bat population is divided among several worker processes, while a distinct process plays the role of the master. Each worker locally executes the steps of the sequential Bat Algorithm on a sub-population, while the master is responsible for global synchronization and for selecting the best solution across the entire population.

The authors describe this behavior using a pseudo-code entitled **Parallel-Distributed Bat Algorithm – Master–Worker (PDBA-MW)**, which highlights the respective responsibilities of the workers and the master.

##### Parallel-Distributed Bat Algorithm – Master–Worker (PDBA-MW)

**Initialization**
- Initialize `every_send_time` to 50

**Main Loop**
- For `iteration = 1` to `Max_iteration`:

  - **If (Worker Node)** then:
    - For `i = 1` to `newpopsize`:
      - *Note*: `newpopsize = N / total_workers`
      - Perform Steps 1 to 5 of the sequential algorithm
    - End For

    - If (`iteration == every_send_time`) then:
      - Each `worker_k` sends its best candidate solution and corresponding fitness to the Master
      - Receive the best solution and fitness from the Master node
    - End If

  - **Else if (Master Node)** then:
    - For `k = 1` to `Total_Workers`:
      1. Receive `best_k` from each worker
      2. Compare and choose the best among all received solutions
      3. Send the best solution to each worker
    - End For

- End For

**Termination Condition**
- The process is terminated when the desired accuracy is achieved  
  or when the maximum number of iterations has been reached.

##### Initialization and Communication Parameterization

The pseudo-code introduces a parameter `every_send_time`, initialized to a fixed value (for example, 50), which controls the communication frequency between the workers and the master. This parameter aims to space out synchronizations between processes in order to limit overly frequent communications that could increase the overall execution cost.

##### Parallel Processing on the Worker Side

At each iteration, each worker executes the Bat Algorithm on a sub-population of size  
`newpopsize = N / total_workers`, where `N` denotes the total population size. As indicated in the pseudo-code, each worker locally applies the main steps of the sequential algorithm (updating positions and velocities, generating candidate solutions, evaluating the objective function, and updating internal parameters).

The operations performed during this phase are executed locally by each worker, without inter-process communication. The independent processing of sub-populations thus allows the computation to be distributed across multiple cluster nodes.

##### Periodic Synchronization with the Master

When the condition `iteration == every_send_time` is satisfied, each worker identifies its best local solution and sends it to the master process, along with the corresponding fitness value. In the implementation proposed by the authors, this step relies on point-to-point communications between the workers and the master, using the MPI `Send/Recv` primitives.

The master then receives the local best solutions sent by the workers, compares them, and extracts the global best solution. This solution is subsequently broadcast to all workers and used as \( x_{\text{best}} \) in the following iterations of the algorithm.

##### Critical Analysis of the Model

The master–worker model introduces a clear and easy-to-implement organization, but it relies on a **centralized global synchronization** around the master process. At each synchronization phase, the master must receive the local best solutions from all workers, perform a global comparison, and redistribute the best solution \( x_{\text{best}} \) to all processes. This organization implies that the communication volume handled by the master increases **linearly with the number of workers**.

When the number of workers becomes large, the master process can thus become a **potential bottleneck**, particularly due to the use of blocking MPI communications such as `Send` and `Recv`, as indicated by the authors. In this context, the time spent in communication may gradually limit the gains obtained from parallelization, even though this situation is not explicitly studied in the presented experiments.

The experimental results nevertheless show an improvement in execution time and an **increasing speed-up** as the cluster size grows, especially for large population sizes. However, the efficiency curves also indicate a gradual decrease in efficiency as the number of nodes increases, which can be interpreted as a combined effect of communication overhead and the load imposed on the master process.

Moreover, the choice of a fixed value for the `every_send_time` parameter reduces synchronization frequency but introduces a **parameter dependency** whose impact on algorithm convergence is not analyzed in detail. Excessively spaced synchronizations may delay the dissemination of \( x_{\text{best}} \), while overly frequent synchronizations may amplify communication costs, particularly in a distributed setting.

Thus, the master–worker model enables a straightforward parallelization of the Bat Algorithm and ensures dissemination of the global best solution to all processes. However, this approach relies on centralized synchronization that may become limiting as the number of workers increases. The communication load borne by the master process then grows with cluster size, potentially affecting the overall scalability of the algorithm.

These limitations have led to the investigation of alternative strategies aimed at reducing communication centralization. Among them, models based on largely independent sub-populations offer a more distributed organization, potentially reducing synchronization costs and improving scalability on larger parallel architectures.

#### Independent Sub-Population Model (Island Model)

The independent sub-population model, also known as the *Island Model*, is based on dividing the global population into several groups that evolve autonomously. This strategy is notably adopted by Tsai et al., who propose a parallelized version of the Bat Algorithm incorporating a periodic communication strategy between sub-populations.

##### Steps of the Sub-Population Model

##### Initialization

The initial bat population is generated and then divided into \( G \) sub-populations. Each subgroup is independently initialized using the Bat Algorithm.

A set of iterations \( R \) is defined to determine when the communication strategy will be applied. The position of a bat is denoted \( X_{ij}^t \), where \( i \) refers to the individual within subgroup \( j \), and \( t \) denotes the current iteration. The algorithm starts with \( t = 1 \).

##### Evaluation

At each iteration, the objective function \( f(X_{ij}^t) \) is evaluated for all bats in each sub-population. This step is performed locally, without any information exchange between groups.

##### Update

Bat velocities and positions are updated using the equations of the original Bat Algorithm. Each sub-population evolves independently and maintains its own local best solution, without direct knowledge of the solutions found by other groups.

##### Communication Strategy

The communication strategy is activated only at specific iterations defined by the set  
\( R = \{ R_1, 2R_1, 3R_1, \dots \} \).

When a communication iteration is reached, each subgroup selects the \( k \) best bats according to their fitness values. These solutions are then copied to the neighboring subgroup \( g_{(p+1) \bmod G} \), where they replace the same number of worst-performing solutions.

This mechanism allows a gradual diffusion of information between sub-populations, without global synchronization or a centralized process.

##### Termination

The previous steps are repeated until a stopping criterion is met, such as a maximum number of iterations or achieving a target value of the objective function. The best value obtained and its corresponding position are then recorded as the final solution.

##### Limitations of the Independent Sub-Population Model

The independent sub-population model proposed by Tsai et al. reduces communication frequency and avoids centralized global synchronization. However, this organization introduces certain limitations from the perspective of algorithmic dynamics.

In particular, the global best solution is not continuously shared among all individuals. Each sub-population evolves with its own local best solution, and information from other groups is only integrated periodically through migrations. This delayed diffusion may slow convergence, especially when some sub-populations explore less promising regions of the search space.

Furthermore, this strategy relies on several additional parameters, such as the number of sub-populations, communication frequency, and the number of migrated solutions. The choice of these parameters strongly influences the balance between exploration and exploitation, without their impact being thoroughly analyzed.

Thus, while this model promotes diversity and scalability, it deviates from the behavior of the sequential Bat Algorithm and introduces significant trade-offs between convergence quality, information diffusion speed, and parameterization complexity.

In the remainder of this work, we propose an approach aimed at leveraging parallelization while limiting modifications to the original functioning of the algorithm.

---

# Our Implementation Strategy

To effectively leverage the available HPC resources (multi-core nodes and distributed clusters), we implemented two parallel versions of the Bat Algorithm:
1. A **Shared Memory** version using **OpenMP**.
2. A **Distributed Memory** version using **MPI**.

Both implementations aim to preserve the exact algorithmic behavior of the sequential version while accelerating the computation of the most intensive parts.

## 1. MPI Implementation (Distributed Memory)

Our MPI implementation follows the **Single Program Multiple Data (SPMD)** paradigm. The total population of $N$ bats is evenly divided among $P$ available MPI processes. This approach is well-suited for distributed clusters where memory is not shared between nodes.

### Initialization and Distribution
- **Rank 0** (the root process) is responsible for initializing the entire population of bats.
- We use `MPI_Scatter` to distribute equal chunks of the population to all processes.
- Each process $k$ receives a local array `local_bats` of size $N/P$ and maintains its own random number generator (seeded with `time + rank` to ensure diversity).

In our final implementation, we made the random number generation **deterministic and comparable** across sequential/OpenMP/MPI:
- Rank 0 initializes the full population using a user-provided `--seed`.
- Each bat stores its own RNG state inside the `Bat` struct.
- That RNG state is scattered together with the bat data, so every rank continues the same bat “random stream”.

This is important for benchmarking: using C's `rand()` in OpenMP can be unsafe (shared global state), and different random paths can change the amount of work performed.

### Main Loop and Synchronization
The core of the algorithm is parallelized as follows:

1.  **Local Update**: Each process updates its subset of bats (velocity, position, frequency, objective function) independently. This phase requires no communication.
2.  **Local Best Finding**: Each process scans its `local_bats` to find the best candidate within its own partition.
3.  **Global Best Reduction**: To identify the global best solution across the entire cluster, we avoid the naive "Master-Worker" bottleneck where one node receives all data. Instead, we use the collective operation `MPI_Allreduce` with the `MPI_MAXLOC` operator.
    - We create a pair structure `{ double val; int rank; }`.
    - `MPI_Allreduce` compares these pairs across all processes and returns the maximum fitness value and the **rank ID** that owns it.
4.  **Broadcast**: Once the "owner" rank of the global best is identified, that specific rank broadcasts the full `Bat` structure (position, velocity, etc.) to all other processes using `MPI_Bcast`.

This design is highly efficient because `MPI_Allreduce` typically uses tree-based algorithms ($\mathcal{O}(\log P)$ complexity), avoiding the congestion of point-to-point communication to a single master node.

## 2. OpenMP Implementation (Shared Memory)

For multi-core shared-memory architectures (single node), we used OpenMP to parallelize the loop over the bat population.

### Parallel Region
- The array `bats` is stored in shared memory, accessible by all threads.
- We use the directive `#pragma omp parallel` to spawn a team of threads.
- The main update loop is distributed using `#pragma omp for`. The loop iterations (indices $0$ to $N-1$) are divided among the threads.

### Handling Race Conditions
A critical challenge in the shared-memory approach is updating the `global_best` solution. If multiple threads find a new best solution simultaneously and try to write to the shared `best_bat` variable, a race condition occurs.

To solve this efficienty:
1.  **Thread-Local Best**: Each thread maintains a private variable `thread_best` initialized to the current best.
2.  **Local Comparison**: Inside the loop, threads update their `thread_best` without locking, avoiding overhead.
3.  **Critical Section**: After the loop finishes, we use a `#pragma omp critical` block. Threads enter this block one by one to compare their `thread_best` with the shared `iter_best` and update it if necessary.

This strategy minimizes synchronization overhead (locks are only acquired once per thread per iteration) while guaranteeing data correctness.

## Summary of Parallelism

| Feature | MPI Version | OpenMP Version |
| :--- | :--- | :--- |
| **Model** | Distributed Memory (Message Passing) | Shared Memory (Threads) |
| **Data Partitioning** | Hard split (Scatter) | Loop splitting (Work-sharing) |
| **Synchronization** | Explicit (`Allreduce`, `Bcast`) | Implicit (End of parallel region) + `critical` |
| **Ideally suited for** | Multiple nodes (Cluster) | Single multi-core node |

---

# Performance and Scalability Analysis

This section evaluates the performance of our implementations in terms of **execution time**, **speedup**, and **efficiency**, as required for HPC benchmarking.

## Metrics

Let $T_1$ be the execution time of the best sequential implementation and $T_p$ the time of a parallel execution using $p$ workers (threads or MPI processes).

- **Execution time**: $T_p$ (wall-clock time).
- **Speedup**: $S(p) = \dfrac{T_1}{T_p}$.
- **Efficiency**: $E(p) = \dfrac{S(p)}{p}$.

For OpenMP we define $p$ as the number of **threads**; for MPI we define $p$ as the number of **processes**.

## Timing Methodology

To obtain reliable measurements, we instrumented each implementation with high-resolution wall-clock timers:

- **Sequential**: `clock_gettime(CLOCK_MONOTONIC, ...)`.
- **OpenMP**: `omp_get_wtime()`.
- **MPI**: `MPI_Wtime()`.

For MPI, we bracket the measured region with `MPI_Barrier(MPI_COMM_WORLD)` to synchronize all ranks before starting and after finishing the timed region. The reported MPI time is the **maximum** elapsed time across ranks (using `MPI_Reduce(..., MPI_MAX, ...)`), which reflects the true end-to-end parallel time.

To minimize measurement noise due to I/O, the codes provide a `--quiet` flag that disables iteration printing during benchmarks.

Because the Bat Algorithm is stochastic, we also fix the seed (e.g., `--seed 1`) when benchmarking so that runs are reproducible.

Each run prints a machine-readable line:

```
BENCH version=<sequential|openmp|mpi> n_bats=<N> iters=<T> procs=<P> threads=<K> time_s=<seconds>
```

These lines are collected and used to compute $S(p)$ and $E(p)$.

## Strong vs Weak Scalability

We evaluate two scalability scenarios:

### Strong Scalability

We keep the problem size constant (fixed number of bats $N$ and iterations $T$) and increase $p$.
This quantifies how fast a fixed problem runs when more resources are added.

### Weak Scalability

We increase the problem size proportionally with $p$ (e.g., keep $N/p$ constant). A method is weakly scalable if its efficiency stays approximately constant as $p$ grows.

## Practical Benchmark Procedure on the Cluster

On the UNITN HPC cluster we submit a PBS job that runs:

- a sequential baseline,
- OpenMP tests for several thread counts,
- MPI tests for several process counts,
- and a weak-scaling campaign where $N$ grows with $p$.

We provide an example PBS script in `code/benchmark.pbs`. The output file produced by PBS contains the `BENCH ...` lines.

## Plot Generation

To produce the requested graphs (time, speedup, efficiency) we use a small parser script:

```
python3 tools/bench_analyze.py --input code/bench_out.txt --outdir bench_out
```

This produces:

- `bench_out/bench_metrics.csv` with both strong and weak scaling metrics.
- A small set of combined comparison plots (sequential vs OpenMP vs MPI), including two baseline choices:
  - **vs sequential baseline**: compares parallel versions to the sequential program.
  - **vs self baseline**: compares MPI to MPI(p=1) and OpenMP to OpenMP(p=1).

In practice, the **self-baseline** plots are often the safest to discuss, because different programs (sequential vs OpenMP vs MPI) may have different overheads even when they implement the same algorithm.

