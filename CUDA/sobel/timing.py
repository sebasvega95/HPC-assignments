from time import time
from os import remove
from matplotlib.image import imread
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def time_a_function(program, args):
    start = time()
    subprocess.call([program] + [args])
    end = time()
    return float(end - start)


def clean(programs):
    for p in programs:
        remove(p)


def plot_results(times, programs, images):
    x = [imread(img)[:,:,0].shape for img in images]
    xlabels = [str(xi) for xi in x]
    x = np.array([np.prod(xi)  for xi in x])
    width = 0.25 * np.min(x)
    colors = ['r', 'g', 'b', 'y']
    for i, (p, c) in enumerate(zip(programs, colors)):
        y = [times[p][img]['mean'] for img in images]
        std_y = [times[p][img]['std'] for img in images]
        plt.bar(x + width * i, y, width, color=c, yerr=std_y, label=p)
    plt.xlabel('Size of image')
    plt.ylabel('Time of Sobel operator (s)')
    plt.legend()
    plt.show()


def print_results(times, programs, images):
    sizes = [imread(img)[:,:,0].size for img in images]
    for p in programs:
        print '\n{}'.format(p)
        print 'Image'.rjust(13), 'Size'.rjust(9), 'Avg. time'.rjust(10), 'Std. time'.rjust(10)
        for img, size in zip(images, sizes):
            mean_t = times[p][img]['mean']
            std_t = times[p][img]['std']
            print '{:13} {:9d} {:10.5f} {:10.5f}'.format(img, size, mean_t, std_t)


def print_csv(times, programs, images):
    sizes = [r'{1}\times{0}'.format(*imread(img)[:,:,0].shape) for img in images]
    program_names = ['CPU', 'OpenCV CPU', 'GPU', 'OpenCV GPU']
    f = open('times.csv', 'w')
    f.write('Version,Image,Size (px$^2$),Average time (s),Error (s)\n')
    for p, pname in zip(programs, program_names):
        first_row = True
        for img, size in zip(images, sizes):
            mean_t = times[p][img]['mean']
            std_t = times[p][img]['std']
            version = pname if first_row else ' '
            first_row = False
            f.write('{},{},${}$,{:.5f},{:.5f}\n'.format(version, img.replace('img/', ''), size, mean_t, std_t))
    f.close()


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
    programs = ['./sobel-seq.out', './cv-sobel-seq.out', './sobel.out', './cv-sobel.out']
    images = ['img/space{}.png'.format(i) for i in range(1, 6)]
    times = {}
    try:
        print 'Loading times.json...'
        time_file = open('times.json', 'r')
        times = json.load(time_file)
        time_file.close()
    except IOError:
        print 'Failed, calculating times'
        times = calculate_time(programs, images)
    print_results(times, programs, images)
    print_csv(times, programs, images)
    plot_results(times, programs, images)
    clean(programs)


if __name__ == '__main__':
    main()

