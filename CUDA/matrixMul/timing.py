from time import time
from os import remove
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def time_a_function(program, arg):
    start = time()
    subprocess.call([program, arg])
    end = time()
    return float(end - start)


def clean(programs):
    for p in programs:
        remove(p)


def plot_results(times, programs, args):
    x = [float(arg) for arg in args]
    for p in programs:
        y = [times[p][arg]['mean'] for arg in args]
        std_y = [times[p][arg]['std'] for arg in args]
        plt.errorbar(x, y, yerr=std_y, fmt='o:')
        plt.xlabel('Image size')
        plt.ylabel('Time (s)')
    plt.show()


def print_results(times, programs, sizes):
    for p in programs:
        print '\n{}'.format(p)
        print 'n'.rjust(4), 'Avg. time'.rjust(10), 'Std. time'.rjust(10)
        for size in sizes:
            mean_t = times[p][size]['mean']
            std_t = times[p][size]['std']
            print '{:4} {:10.5f} {:10.5f}'.format(size, mean_t, std_t)


def calculate_time(programs, args):
    times = {}
    num_runs = 20
    for p in programs:
        times[p] = {}
        for arg in args:
            times[p][arg] = {}
            t = []
            print 'Running {} with {} {} times...'.format(p, arg, num_runs),
            for _ in range(num_runs):
                t.append(time_a_function(p, arg))
            mean_t = np.mean(t)
            std_t = np.std(t)
            print '({} +- {})s on average'.format(mean_t, std_t)
            times[p][arg]['mean'] = mean_t
            times[p][arg]['std'] = std_t
    time_file = open('times.json', 'w')
    print 'Writing times.json...'
    json.dump(times, time_file)
    return times


def main():
    print 'Running make...'
    subprocess.call(['make', '-j8'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    programs = ['./matrixMul.out', './matrixMul-seq.out']
    sizes = ['100', '500', '1000', '2000', '2500']
    times = {}
    try:
        print 'Loading times.json...'
        time_file = open('times.json', 'r')
        times = json.load(time_file)
        time_file.close()
    except IOError:
        print 'Failed, calculating times'
        times = calculate_time(programs, sizes)
    print_results(times, programs, sizes)
    plot_results(times, programs, sizes)
    clean(programs)


if __name__ == '__main__':
    main()

