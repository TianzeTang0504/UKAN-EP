import multiprocessing

def cpu_bound_task(n):
    count = 0
    for i in range(n):
        count += i
    return count

if __name__ == "__main__":
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=cpu_bound_task, args=(100000000,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
