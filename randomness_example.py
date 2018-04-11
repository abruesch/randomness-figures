#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np



# data structures used for data storage later on.
sum_distribution = []

transition_probs = {x: 0 for x in range(0, 128)}
transition_count = [0]

# generate some random data
keys = [ format(random.SystemRandom().randint(0,2**128),'#0128b')[2:] for x in range(0,1411)]

def random_walk(bits):
    ''' takes string of '0' and '1' and turns them into random walks in a galton board. Effectively computes
	the cumulative sums distribution for every prefix of the input string. '''
    transition_count[0] += 1
    ret = [0]
    for i, b in enumerate(bits[:128]):
        val = -1
        if b == '1':
            val = 1
            transition_probs[i] += 1
        ret.append(ret[-1] + val)

    sum_distribution.append(ret[-1])


def markov_transitions():
    ''' transition probabilities from every nth to (n+1)th bit. '''
    norm_transition_probs = [transition_probs[x] / transition_count[0]
                             for x in range(0, 128)]
    plt.clf()
    plt.xlabel('nth bit')
    plt.ylabel('Probability for 1')
    plt.ylim([0, 1])
    plt.xlim([0, 128])
    plt.plot(norm_transition_probs,color='b')
    plt.savefig('figures/example_markov.png',dpi=300,orientation='landscape')

from scipy.misc import comb

def binomial_plot(n=256):
    ''' Theoretical binomial distribution that is plotted as red line into figure.'''
    x = np.arange(256)
    y = list(map(lambda xi: comb(256,xi)/2**n, x))
    x = list(map(lambda r: r-2*63.25, x))
    plt.plot(x,y,color='r')

#def binomial_plot(bins,n):
#    variance =  n * 0.5 * 0.5
#    sigma = np.sqrt(variance)
#    y = mlab.normpdf(np.asarray(list(range(-128,128))), 0, sigma)
#    plt.plot(np.asarray(list(range(-128,128))),y,color='r')

def distribution():
    ''' Plots the cumulative sums distribution and saves figure. '''
    plt.clf()
    count, bins, ignored = plt.hist(sum_distribution, color='#007a9b', range=(-128,128), normed=True,rwidth=0.5, bins=128)
    binomial_plot(256)
    #binomial_plot(len(count))
    plt.savefig('figures/example_distribution.svg',dpi=300,orientation='landscape')


def plot_rand_walk():
    ''' turns string consisting of '0' and '1' into random walks. While the walks are currently not plotted, the distribution
	of the cumulative sums along with the markov transitions are computed and plotted using them. results are currently
	passed via vars in scope.'''
    from cycler import cycler
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['#e78a33', '#eda766', '#8d4959', '#aa7782', '#bdcd61', '#cdda89',
                                                     '#8a9c33', '#a7b566'])

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('font', serif='cm')
    matplotlib.rcParams['font.size'] = 18.0
    plt.ylabel('Sum')
    plt.xlabel('Keylength')

    plt.ylim([-128, 128])
    for key in keys:
        random_walk(key)
    plt.tight_layout()
    distribution()
    markov_transitions()


def plot_heat_map(pt=plt):
    ''' creates heatmap of random walks. Uses the 'keys' var
	in scope as input.'''
    intensity = 1
    heatmap_array = []
    for key in keys:
        val = int(key[0])
        for i, bit in enumerate(key[1:128]):
            if bit == '1':
                val = val + 1
            elif bit == '0':
                val = val - 1
            heatmap_array.append([i, val])

    np_srt_heat = np.asarray(heatmap_array)
    X = np.take(np_srt_heat, [0], axis=1).flatten()
    Y = np.take(np_srt_heat, [1], axis=1).flatten()
    # bins = (range(max(X)), range(min(Y),max(Y)))
    bins = (range(128), range(min(Y), max(Y)))
    H, xedges, yedges = np.histogram2d(X, Y, bins=bins, normed=False)
    # H = H.T
    # H = H
    H = intensity * H
    # 'viridis'
    pt.xlabel('Sum')
    pt.ylabel('Keylength')
    pt.imshow(H, cmap='viridis', norm=matplotlib.colors.LogNorm(), interpolation='nearest', origin='upper',
              extent=[yedges[0], yedges[-1], xedges[-1], xedges[0]])
    plus_y = [i for i in range(-1, 127)]
    plus_x = [i for i in range(1, 129)]
    minus_y = [126 - i for i in range(0, 128)]
    minus_x = [-127 + i for i in range(1, 129)]
    plt.tight_layout()
    pt.plot(plus_x, plus_y, color='r')
    pt.plot(minus_x, minus_y, color='r')


def apply_plot_heat_map():
    ''' Calls plot_heat_map() and additionally saves the plotted figure.'''
    plot_heat_map()
    plt.savefig('figures/heatmapExample.png', dpi=300, orientation='landscape')


if __name__ == "__main__":
    # layout stuff
    import matplotlib
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FuncFormatter

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('font', serif='cm')

    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['font.size'] = 12.0

    plt.gcf().set_rasterized(True)

    # argument handling
    if sys.argv[1] == 'heatmap':
        apply_plot_heat_map()
    elif sys.argv[1] == 'rand':
        plot_rand_walk()
