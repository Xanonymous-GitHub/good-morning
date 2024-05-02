import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import random, sample
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

nodes: Final[tuple[int, ...]] = (100, 200, 400, 600, 800, 1000)
alpha: Final[float] = 0.1
init_ratio_of_one: Final[float] = 0.4
spread_repeat_times: Final[int] = 500
opinion_update_times_limit: Final[int] = 2000000


def initialize_opinions(*, size: int, ratio_of_one: float) -> np.ndarray:
    return np.random.choice([0, 1], size, p=[1 - ratio_of_one, ratio_of_one])


def opinion_update(
        *,
        current_opinions: np.ndarray,
        all_graph_members: tuple[int, ...]
) -> None:
    size = current_opinions.size
    target_opinion = int(random() * size)

    if random() <= alpha:
        current_opinions[target_opinion] = 1
        return

    chosen_neighbors = sample(
        all_graph_members[:target_opinion] + all_graph_members[target_opinion + 1:],
        2
    )

    if current_opinions[chosen_neighbors].sum(axis=0) + current_opinions[target_opinion] >= 2:
        current_opinions[target_opinion] = 1
    else:
        current_opinions[target_opinion] = 0


def update_opinion_until_all_one(
        current_opinions: np.ndarray,
        all_graph_members: tuple[int, ...]
) -> int:
    current_opinions = current_opinions.copy()
    times = 0

    while current_opinions.sum() != current_opinions.size:
        opinion_update(
            current_opinions=current_opinions,
            all_graph_members=all_graph_members
        )
        times += 1
        # print(f"\033[H\033[JN is {current_opinions.size}, Iteration {times}")
        if times >= opinion_update_times_limit:
            print(f"Exceed the limit of opinion update times, retried...")
            current_opinions = current_opinions.copy()
            times = 0

    return times


def avg_opinion_update_times_of(size: int) -> float:
    if size == 0:
        return 0.0

    all_graph_members = tuple(range(size))

    with ProcessPoolExecutor() as executor:
        missions = [
            executor.submit(
                update_opinion_until_all_one,
                initialize_opinions(size=size, ratio_of_one=init_ratio_of_one),
                all_graph_members
            )
            for _ in range(spread_repeat_times)
        ]

        opinion_update_times = [
            future.result()
            for future in as_completed(missions)
        ]

    return sum(opinion_update_times) / (spread_repeat_times * size)


def start() -> None:
    avg_opinion_update_times: Final[list[float]] = []

    for n in nodes:
        print(f"N is {n}, please wait...")
        result = avg_opinion_update_times_of(n)
        avg_opinion_update_times.append(result)
        print(f"Average opinion update times of {n} nodes: {result}\n")

    log_function = 1.5 * np.log(nodes)

    # Plotting the functions
    plt.figure(figsize=(10, 5))
    plt.plot(nodes, avg_opinion_update_times, label='Simulation')
    plt.plot(nodes, log_function, label='1.5 * ln(n)', linestyle='--')
    plt.title('Comparison of t_n_p and 1.5ln(n)')
    plt.xlabel('n')
    plt.ylabel('t_n_p')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis line
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis line
    plt.show()


if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
