import sys
from collections.abc import Generator
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)
from multiprocessing import current_process, cpu_count, Queue
from random import random, sample
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

nodes: Final[tuple[int, ...]] = (100, 200, 400, 800)
alpha: Final[float] = 0.1
init_ratio_of_one: Final[float] = 0.4
spread_repeat_times: Final[int] = 500
opinion_update_times_limit: Final[int] = 10000000

# The shared queue for the tasks of the processes
# Type: tuple[int, tuple[Callable[[int, int, list[int]], list[float]], int, int, list[int]]]
shared_process_tasks: Final[Queue] = Queue()


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


def submit_opinion_updates_of(size: int) -> None:
    batch_size_per_process = (
        spread_repeat_times // cpu_amount
        if spread_repeat_times > (cpu_amount := cpu_count()) else 1
    )
    non_handled_size = spread_repeat_times % cpu_amount

    all_graph_members = list(range(size))

    for i in range(cpu_amount):
        shared_process_tasks.put((
            size,
            (
                updates_in_single_process,
                size,
                batch_size_per_process + 1 if i < non_handled_size else batch_size_per_process,
                all_graph_members
            )
        ))

    print(f"N is {size}, please wait...")


def consume_tasks() -> Generator[tuple[int, Future[list[float]]], None, None]:
    with ProcessPoolExecutor() as _p_executor:
        while not shared_process_tasks.empty():
            opinion_size, task = shared_process_tasks.get_nowait()
            yield opinion_size, _p_executor.submit(*task)


def start() -> None:
    for n in nodes:
        submit_opinion_updates_of(n)

    print("All tasks have been submitted, please wait...")

    avg_opinion_update_times: Final[dict[int, list[float]]] = {
        n: []
        for n in nodes
    }

    processing_update_packs: Final[dict[int, list[Future[list[float]]]]] = {
        f: []
        for f in nodes
    }

    for n, future in consume_tasks():
        processing_update_packs[n].append(future)

    with ProcessPoolExecutor() as _p_executor:
        for n, futures in processing_update_packs.items():
            for future in futures:
                avg_opinion_update_times[n].extend(future.result())

    avg_opinion_update_times_results: Final[list[float]] = []
    for n, times in avg_opinion_update_times.items():
        result = sum(times) / (spread_repeat_times * n)
        avg_opinion_update_times_results.append(result)
        print(f"Average opinion update times of {n} nodes: {result}")

    log_function = 1.5 * np.log(nodes)

    # Plotting the functions
    plt.figure(figsize=(10, 5))
    plt.plot(nodes, avg_opinion_update_times_results, label='Simulation')
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
