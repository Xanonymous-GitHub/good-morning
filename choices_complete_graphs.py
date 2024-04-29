# This is a coding experiment using 2-choice updating rule for complete graphs

import asyncio
import concurrent.futures
import sys
from functools import lru_cache
from typing import Final

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@lru_cache(typed=True)
def create_complete_graph(_n: int, /) -> nx.Graph:
    """Create a d-regular graph with n nodes"""
    return nx.complete_graph(_n)


@lru_cache(typed=True)
def initialize_opinions(_n: int, _p: float, /) -> np.ndarray:
    # Create an array of zeros
    opinions = np.zeros(_n, dtype=np.uint8)
    # Calculate the number of ones based on the proportion p
    num_ones = int(_n * _p)
    # Randomly select num_ones indices to set to 1
    ones_indices = np.random.choice(_n, num_ones, replace=False)
    opinions[ones_indices] = 1
    return opinions


# We define the 2-choice opinion updating rule
def opinion_update(
        graph: nx.Graph,
        current_opinions: np.ndarray,
        _n: int,
        _alpha: float,
        /
) -> np.ndarray:
    # new_opinions = np.copy(current_opinions)

    # Randomly select one node to update its opinion
    i = np.random.randint(_n)
    neighbors = list(graph.neighbors(i))

    if np.random.rand() <= _alpha:  # With alpha probability, change to 1
        current_opinions[i] = 1
        # print(f"Update node {i} with alpha")
        return current_opinions

    # With 1-alpha probability, select 2 neighbors and adopt the majority
    chosen_neighbors = np.random.choice(neighbors, 2, replace=False)

    # Majority is 1 or it's a tie
    if np.sum(current_opinions[chosen_neighbors]) + current_opinions[i] >= 2:
        current_opinions[i] = 1
    else:  # Majority is 0
        current_opinions[i] = 0
    # print(f"Update node {i} with 1-alpha, chosen neighbors = {chosen_neighbors}")
    return current_opinions


async def simulate_opinion_dynamics(_n: int, _p: float, /):
    graph = create_complete_graph(_n)
    # Initial opinions
    count = 1

    opinions = initialize_opinions(_n, p)

    # Simulation
    while np.sum(opinions) != _n:
        opinions = opinion_update(graph, opinions, _n, alpha)
        print(f"\033[H\033[JN is {_n}, Iteration {count}")
        count += 1
    iters.append(count)


async def foo(_n: int, _t_n_p: list[float]):
    await asyncio.gather(*[
        simulate_opinion_dynamics(_n, p)
        for _ in range(500)
    ])

    _t_n_p.append(sum(iters) / len(iters) / _n)
    return _t_n_p


def foo_sync(_n: int, _t_n_p: list[float]):
    return asyncio.run(foo(_n, _t_n_p))


def start() -> None:
    t_n_p: list[float] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = [executor.submit(foo_sync, n, t_n_p) for n in n_list]
        for future in concurrent.futures.as_completed(future_results):
            results = future.result()
            t_n_p.extend([f for f in results])

    n = np.array(n_list)
    log_function = 1.5 * np.log(n)

    # Plotting the functions
    plt.figure(figsize=(10, 5))
    plt.plot(n_list, t_n_p, label='Simulation')
    plt.plot(n_list, log_function, label='1.5 * ln(n)', linestyle='--')
    plt.title('Comparison of t_n_p and 1.5ln(n)')
    plt.xlabel('n')
    plt.ylabel('t_n_p')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis line
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis line
    plt.show()


n_list: Final[tuple[int, ...]] = (100, 200, 400, 600, 800, 1000)  # Number of nodes

alpha: Final[float] = 0.1  # Bias towards superior opinion
p: Final[float] = 0.35  # Initial proportion of superior opinion

iters: Final[list[int]] = []

if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
