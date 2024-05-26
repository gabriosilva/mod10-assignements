from threading import Thread
import time
import queue


def sum_slice(numbers_slice, results_queue):
    """
    Calculate the sum of the list `numbers_slice` and store the result in the `results_queue`.

    Args:
        numbers_slice (list): Slice of the list of numbers to sum.
        results_queue (queue.Queue): Queue to store the result of the sum.
    """
    result = sum(numbers_slice)
    results_queue.put(result)


def create_threads(numbers, results_queue, num_threads):
    """
    Create and return a list of threads for summing slices of `numbers`.

    Args:
        numbers (list): List of numbers to sum.
        results_queue (queue.Queue): Queue to store results of the sums.
        num_threads (int): Number of threads to create.

    Returns:
        list: List of threads.
    """
    threads = []
    slice_size = len(numbers) // num_threads  # Calculate slice size
    for i in range(num_threads):
        start = i * slice_size
        # Adjust the end for the last slice to cover the remainder
        end = (i + 1) * slice_size if i != num_threads - 1 else len(numbers)
        # Create a thread with the sliced portion of the list
        thread = Thread(target=sum_slice, args=(
            numbers[start:end], results_queue))
        threads.append(thread)  # Append thread to list
    return threads


def main():
    """
    Main function to sum lists of different sizes using multi-threading and single-threading,
    and print the results and execution time for each method.
    """
    # Different sizes of number lists to test
    numbers_list_sizes = [100, 100000, 1000000]
    numbers = [list(range(1, size + 1)) for size in numbers_list_sizes]

    num_threads = 2  # Number of threads to use for parallelism

    for num_list in numbers:
        results_queue = queue.Queue()  # Queue for results

        # Multi-threading sum calculation
        start_time = time.time()
        threads = create_threads(num_list, results_queue, num_threads)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Collect results from the queue
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        total = sum(results)
        end_time = time.time()
        print(f"Calculated total with threading: {total}")
        print(f"Time to calculate the sum of list with size {
              len(num_list)}: {end_time - start_time} seconds (Multi-thread)")

        # Single-thread sum calculation
        start_time = time.time()
        total_single_thread = sum(num_list)
        end_time = time.time()
        print(f"Expected total: {total_single_thread}")
        print(f"Time to calculate the sum of list with size {len(num_list)}: {
              end_time - start_time} seconds (Single-thread)")


if __name__ == '__main__':
    main()
