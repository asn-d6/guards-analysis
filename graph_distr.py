"""
File to make the graph of the probability distribution
"""


import analysis
import sys
import os
import logging

import util

import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def init_graph_style():
    sns.set_color_palette("deep", desat=.6)
    mpl.rc("figure", figsize=(8, 4))
    np.random.seed(9221999)

def main():
    # Print usage if corrupted CLI
    if len(sys.argv) < 3:
        print "usage: %s <consensus file> <speed threshold in kB>" % sys.argv[0]
        print "example: %s cached-consensus 200" % sys.argv[0]
        sys.exit(1)

    # Get speed threshold and consensus filename from CLI
    speed_threshold = int(sys.argv[2])
    consensus_fname = sys.argv[1]

    # Initialize matplotlib/seaborn/etc.
    init_graph_style()

    # Parse the consensus
    guards = util.parse_consensus(consensus_fname)

    # Consensus parsing is done: calculate all the guard attributes.
    guards.update_guard_attributes()

    # Get the original probability distribution.
    original_prob_distr = guards.get_prob_distr()

    # Prune the guard list (remove slow guards)
    guards.prune_guards(speed_threshold)

    # Pruning is done: recalculate the guard attributes.
    guards.update_guard_attributes()

    # Get the original probability distribution.
    pruned_prob_distr = guards.get_prob_distr()

    # Create uniform distribution too (for comparison)
    # We use the size of the original probability distribution since
    # it's the best case scenario (more guards)
    size = len(original_prob_distr)
    mean = 1/float(size)
    uniform = [mean]*size
    logging.debug("Uniform distribution with mean = %s", mean)

    # Plot it!
    distrs = (uniform, original_prob_distr, pruned_prob_distr)
    labels = ("uniform probability distribution (best case) [%d guards]" % len(original_prob_distr),
              "original probability distribution [%d guards]" % len(original_prob_distr),
              "probability distribution after pruning at %s kB/s [%d guards]" % (speed_threshold, len(pruned_prob_distr)))
#   sns.violinplot(distrs, names=labels, inner="points", color="Set3")
    sns.boxplot(distrs, names=labels, color="Set2")
#    sns.kdeplot(original_prob_distr, cumulative=True)


    plt.show()

if __name__ == '__main__':
#    logging.getLogger("").setLevel(logging.DEBUG)
    main()

