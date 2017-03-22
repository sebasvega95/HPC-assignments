from time import time
from subprocess import call
from os import remove
from matplotlib.image import imread
import json
import numpy as np
import matplotlib.pyplot as plt


def time_a_function(program, args):
    start = time()
    call([program] + [args])
    end = time()
    return float(end - start)
    

def clean(programs):
    for p in programs:
        remove(p)


def plot_results(times, programs, images):
    x = [imread(img)[:,:,0].shape for img in images]
    xlabels = [str(xi) for xi in x]
    x = [np.prod(xi)  for xi in x]
    for p in programs:
        y, std_y = zip(*times[p])
        # plt.plot(x, y, 'o')
        plt.errorbar(x, y, yerr=std_y, fmt='o')
        plt.xticks(x, xlabels)
        plt.xlabel('Image size')
        plt.ylabel('Time (s)')
    plt.show()


def print_results(times, programs, images):
    sizes = [imread(img)[:,:,0].size for img in images]
    for p in programs:
        print '\n{}'.format(p)
        mean_t, std_t = zip(*times[p])
        print 'Image'.rjust(13), 'Size'.rjust(8), 'Avg. time'.rjust(10), 'Std. time'.rjust(10)
        for img, size, m, s in zip(images, sizes, mean_t, std_t):
            print '{:13} {:8d} {:10.5f} {:10.5f}'.format(img, size, m, s)


def main():
    print 'Running make...'
    call(['make', '-j8'])
    print '\n'
    programs = ['./grayscale.out', './grayscale-seq.out']
    images = ['img/emma{}.png'.format(i) for i in range(1, 6)]
    n = 20
    times = {}
    
    try:
        print 'Loading times.json...'
        time_file = open('times.json', 'r')
        times = json.load(time_file)
    except IOError:
        print 'Failed, calculating times'
        for p in programs:
            times[p] = []
            for img in images:
                t = []
                print 'Running {} with {} {} times...'.format(p, img, n),
                for _ in range(n):
                    t.append(time_a_function(p, img))
                mean_t = np.mean(t)
                std_t = np.std(t)
                print '({} +- {})s on average'.format(mean_t, std_t)
                times[p].append((mean_t, std_t))
        time_file = open('times.json', 'w')
        print 'Writing times.json...'
        json.dump(times, time_file)
    time_file.close()
    print_results(times, programs, images)
    plot_results(times, programs, images)
    clean(programs)


if __name__ == '__main__':
    main()

