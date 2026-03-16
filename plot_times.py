#!/usr/bin/env python3
import subprocess
import os
import matplotlib.pyplot as plt 
import numpy as np
import re

nr_threads_list = [2, 4, 8]

def run_and_plot(exec_name: str, start_range: int, end_range: int):
    print(f"Benchmarking: {exec_name}")
    
    time_serial = []
    time_parallel = []
    N_list = []

    ## SERIAL + PARALLEL with one one thread
    for i in range(start_range, end_range):
        N = i
        N_list.append(N)
        
        # use one thread
        my_env = os.environ.copy()
        my_env["ONLY_SERIAL"] = "true"
        if "ONLY_PARALLEL" in my_env:
            del my_env["ONLY_PARALLEL"]

        result = subprocess.run([exec_name, str(N)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
        output = result.stdout.decode('utf-8')
        for line in output.splitlines():
            if "serial" in line:
                # print(line, times)
                times = re.findall(r"[-+]?\d*\.\d+|\d+ s", line)
                t = str(times[1]) # elapsed_real_time
                # t = str(times[0]) # elapsed_cpu_time
                time_serial.append(float(t))
                print(f"{N};{2**N};{exec_name};serial;1;{t}", flush=True)

    ## PARALLEL with multiple threads
    for n_threads in nr_threads_list:
        my_env = os.environ.copy()
        # set the number of threads
        my_env["OMP_NUM_THREADS"] = str(n_threads)
        # don't run serial code and test code
        my_env["ONLY_PARALLEL"] = "true"
        if "ONLY_SERIAL" in my_env:
            del my_env["ONLY_SERIAL"]

        time_parallel_with_n_threads = []
        for i in range(start_range, end_range):
            N = i
            result = subprocess.run([exec_name, str(N)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
            output = result.stdout.decode('utf-8')
            for line in output.splitlines():
                print(line)
                if "parallel" in line:
                    print(line)
                    times = re.findall(r"[-+]?\d*\.\d+|\d+ s", line)
                    t = str(times[1]) # elapsed_real_time
                    time_parallel_with_n_threads.append(float(t))
                    print(f"{N};{2**N};{exec_name};parallel;{n_threads};{t}", flush=True)

        time_parallel.append(time_parallel_with_n_threads)

    # print(len(time_serial))
    # print(len(time_parallel))
    # print(list(map(len, time_parallel)))
    
    ## plot the results
    plt.plot(N_list, time_serial, label="Serial")
    for i in range(len(nr_threads_list)):
        plt.plot(N_list, time_parallel[i], label=f"Parallel {nr_threads_list[i]} threads")

    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'plot_{os.path.basename(exec_name).split(".")[0]}.png', dpi=200, bbox_inches='tight')
    plt.clf() 
if __name__ == "__main__":
    run_and_plot("./bubble.run", 2, 14)
    run_and_plot("./mergesort.run", 10, 28)
    run_and_plot("./odd-even.run", 10, 19)
