# Multi-Armed Bandit Algorithm Comparison

This repository contains code to compare seven different algorithms for solving the Multi-Armed Bandit (MAB) problem using a simulated email-conversion dataset. The primary goal is to illustrate how each strategy balances exploration and exploitation, and to visualize their cumulative regret over time.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Data](#data)
* [Algorithms Implemented](#algorithms-implemented)
* [Installation](#installation)
* [Usage](#usage)
* [Results & Visualizations](#results--visualizations)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

In the classic Multi-Armed Bandit problem, an agent sequentially chooses among $K$ “arms” (e.g., different email variants) to maximize cumulative reward (e.g., click‐through or conversion). After each pull, the agent observes only the reward of the chosen arm. This repository implements seven well‐known MAB strategies, computes their cumulative regret (relative to a known best conversion rate), and produces comparison plots and histograms to illustrate how each algorithm performs over a fixed horizon.

Specifically, we assume a known best conversion rate:

```
BEST_P = 0.213
```

and define regret at time $t$ as

$$
\text{Regret}(t) \;=\; \text{BEST\_P} \times t \;-\; \sum_{s=1}^t \text{(observed reward at step }s).
$$

---

## Data

* **emails.csv**: A CSV file where each row corresponds to a customer (time step) and each column corresponds to a distinct email variant (i.e., an “arm”).

  * Value `1` indicates a successful conversion when that email is sent.
  * Value `0` indicates no conversion.

Place `emails.csv` in the same directory as the main script (or notebook) prior to running.

---

## Algorithms Implemented

1. **Pure Exploration**

   * Randomly selects an arm at each time step.
   * No prior information is used.
   * Starts with zero successes for all arms.

2. **Pure Exploitation (with Prior Successes)**

   * Initializes each arm with one pseudo-success (i.e., `successes[a] = 1`, `counts[a] = 1` for all $a$).
   * At each step, picks the arm with the highest empirical success rate:

     $$
       \hat{p}_a = \frac{\text{successes}[a]}{\text{counts}[a]}.
     $$

3. **Explore‐then‐Exploit (with Prior Successes)**

   * For the first $T_0$ steps (default $T_0 = 100$), selects uniformly at random.
   * After $T_0$, always picks the arm with the highest empirical success rate (same initialization as Pure Exploitation).

4. **Epsilon‐Greedy (with Prior Successes)**

   * At time $t$, sets

     $$
       \varepsilon = \frac{1}{t}.
     $$
   * With probability $\varepsilon$, explores (chooses an arm uniformly at random); otherwise, exploits (chooses the arm with the highest empirical rate).
   * Each arm starts with one initial success and count.

5. **Upper Confidence Bound (UCB, with Prior Successes)**

   * Each arm starts with one pseudo-success and one count (so $\hat{p}_a = 1$ initially).
   * At time $t$, computes for each arm $a$:

     $$
       \text{UCB}_a(t) = \hat{p}_a + \alpha \sqrt{\frac{\ln t}{\text{counts}[a]}}, 
     $$

     where $\alpha = 0.75$ (tunable).
   * Chooses the arm with the largest UCB value.

6. **Thompson Sampling (Beta‐Bernoulli)**

   * Models each arm’s conversion probability as a $\mathrm{Beta}(\alpha_a, \beta_a)$ distribution, initializing $\alpha_a = 1$, $\beta_a = 1$.
   * At each step, samples

     $$
       \theta_a \sim \mathrm{Beta}(\alpha_a, \beta_a) 
     $$

     for each arm; selects the arm with the highest $\theta_a$.
   * Updates $\alpha_a\leftarrow \alpha_a + 1$ if reward = 1, or $\beta_a\leftarrow \beta_a + 1$ if reward = 0.

7. **Bootstrap (with Prior Successes)**

   * Maintains a list of observed rewards for each arm, initialized to $[1]$ (one pseudo-success).
   * At each step, for each arm $a$, draws a bootstrap sample (with replacement) from its reward history, computes the sample mean $\hat{p}_a^{(\text{boot})}$, and chooses the arm with the highest bootstrapped estimate.
   * Observes the reward and appends it to that arm’s history.

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/mab-algo-comparison.git
   cd mab-algo-comparison
   ```

2. **Create a Python virtual environment (optional, but recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` is not yet provided, install:

   ```bash
   pip install numpy pandas matplotlib
   ```

4. **Ensure `emails.csv` is present**

   * Download or copy `emails.csv` into the repository’s root directory.
   * Each row = one customer; each column = one email variant (arm).

---

## Usage

There are two primary entry points:



1. **From the Jupyter Notebook**
   If you prefer interactive exploration, open:

   ```bash
   jupyter notebook MAB_Algorithms_Comparison.ipynb
   ```

   * Run all cells in order.
   * The notebook walks you through all seven algorithms, prints final rewards/regrets, and visualizes:

     * A combined plot of cumulative regret over $t = 1,\dots,T$.
     * Histograms of chosen‐arm frequencies for the first 100 and last 100 customers (for UCB, Thompson Sampling, and Bootstrap).

---

## Results & Visualizations

Running all algorithms for $T = 10{,}000$ (random seed 42) on a typical `emails.csv` yields (example output):

```
=== Pure Exploration ===
Final reward: 1649.00
Final regret: 481.00

=== Pure Exploitation (prior successes) ===
Final reward: 1987.00
Final regret: 143.00

=== Explore then Exploit (T0=100, prior successes) ===
Final reward: 2112.00
Final regret: 18.00

=== Epsilon‐Greedy (epsilon = 1/t, prior successes) ===
Final reward: 1988.00
Final regret: 142.00

=== UCB (alpha=0.75, prior successes) ===
Final reward: 1999.00
Final regret: 131.00

=== Thompson Sampling (Beta‐Bernoulli, uniform prior) ===
Final reward: 2062.00
Final regret: 68.00

=== Bootstrap (with prior successes) ===
Final reward: 2082.00
Final regret: 48.00
```

1. **Cumulative Regret Plot**
   A single plot displaying 7 curves:

   * Pure Exploration
   * Pure Exploitation
   * Explore‐then‐Exploit
   * Epsilon‐Greedy
   * UCB
   * Thompson Sampling
   * Bootstrap
     shows how regret accumulates over time (lower is better).

2. **Histograms of Chosen Arms**
   For UCB, Thompson Sampling, and Bootstrap:

   * **First 100 Customers**: Shows how frequently each email variant was selected early on (emphasis on exploration).
   * **Last 100 Customers**: Shows how the algorithm has converged (likely heavily favoring the arm with highest empirical conversion).

---

## Project Structure

```
mab-algo-comparison/
├── MAB_Algorithms_Comparison.ipynb   ← Jupyter notebook with code & visualizations 
├── emails.csv                       ← Simulated email‐conversion dataset (K columns, N rows)
├── requirements.txt                 ← Lists Python dependencies (numpy, pandas, matplotlib)
├── README.md                        ← This readme file
└── LICENSE                          ← (optional) License file
```

* **`MAB_Algorithms_Comparison.ipynb`**

  * Contains all function definitions (one for each algorithm), a `run_and_compare_algorithms()` driver, and plotting routines.
  * Prints final rewards/regrets and displays comparison plots.


* **`emails.csv`**

  * Each column is an “arm” (email variant).
  * Each row is a separate trial (customer).
  * Contains only `0` or `1` entries.

* **`requirements.txt`**

  * Pins versions of `numpy`, `pandas`, and `matplotlib` (and `jupyter` if needed).

---

## Dependencies

* **Python 3.7+**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **(Optional) Jupyter Notebook**

To install:

```bash
pip install numpy pandas matplotlib
```

If using `run_mab_algos.py` and you want to pin versions, see `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Contributing

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes (e.g., add a new bandit algorithm, improve visualization).
4. Ensure all existing code still runs (`python run_mab_algos.py`).
5. Commit your changes and push (`git push origin feature/your-feature`).
6. Open a Pull Request describing your proposed enhancements.

---

## License

This project is licensed under the MIT License. Feel free to reuse or modify as long as you include attribution. If you use it in academic work, please cite accordingly.

```text
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```

> **Note:** If you prefer a different license (e.g., Apache 2.0, GPLv3), update `LICENSE` accordingly.

---

*Happy exploring the exploration–exploitation trade‐off!*
