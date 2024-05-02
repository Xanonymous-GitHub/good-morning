import sys
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)
from multiprocessing import current_process, cpu_count
from random import random, sample
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

nodes: Final[tuple[int, ...]] = (100, 200, 400, 800)
alpha: Final[float] = 0.1
init_ratio_of_one: Final[float] = 0.4
spread_repeat_times: Final[int] = 500
opinion_update_times_limit: Final[int] = 10_000_000


def initialize_opinions(*, size: int) -> np.ndarray:
    return np.random.choice([0, 1], size, p=[1 - init_ratio_of_one, init_ratio_of_one])


def opinion_update(
        *,
        current_opinions: np.ndarray,
        all_graph_members: list[int]
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
        all_graph_members: list[int]
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


def updates_in_single_process(
        opinion_size: int,
        batch_size: int,
        all_graph_members: list[int]
) -> list[float]:
    process_name = current_process().name
    print(f"{process_name} is working...")
    with ThreadPoolExecutor() as _executor:
        _missions = [
            _executor.submit(
                update_opinion_until_all_one,
                initialize_opinions(size=opinion_size),
                all_graph_members
            )
            for _ in range(batch_size)
        ]

        return [
            future.result()
            for future in as_completed(_missions)
        ]


def avg_opinion_update_times_of(size: int) -> float:
    if size == 0:
        return 0.0

    all_graph_members: list[int] = list(range(size))

    batch_size_per_process = (
        spread_repeat_times // cpu_amount
        if spread_repeat_times > (cpu_amount := cpu_count()) else 1
    )
    non_handled_size = spread_repeat_times % cpu_amount

    print(f"N is {size}, please wait...")

    with ProcessPoolExecutor() as _p_executor:
        missions = [
            _p_executor.submit(
                updates_in_single_process,
                size,
                batch_size_per_process + 1 if i < non_handled_size else batch_size_per_process,
                all_graph_members
            )
            for i in range(cpu_amount)
        ]

        opinion_update_times = [
            times
            for future in as_completed(missions)
            for times in future.result()
        ]

    result = sum(opinion_update_times) / (spread_repeat_times * size)
    print(f"Average opinion update times of {size} nodes: {result}\n")
    return result


def start() -> None:
    avg_opinion_update_times: Final[dict[int, float]] = {
        n: 0.0
        for n in nodes
    }

    with ProcessPoolExecutor() as executor:
        missions: Final[dict[int, Future]] = {
            k: executor.submit(avg_opinion_update_times_of, k)
            for k in avg_opinion_update_times.keys()
        }

        for k, future in missions.items():
            avg_opinion_update_times[k] = future.result()

    log_function = 1.5 * np.log(nodes)

    # Plotting the functions
    plt.figure(figsize=(10, 5))
    plt.plot(nodes, tuple(avg_opinion_update_times.values()), label='Simulation')
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
