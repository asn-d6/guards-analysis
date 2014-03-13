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

THRESHOLD_MAX = 10000
THRESHOLD_STEP = 100

def init_graph_style():
    sns.set(palette="Set2")
    mpl.rc("figure", figsize=(2, 4))
    np.random.seed(9221999)

def do_analysis(guards):
    # List of cutoffs
    cutoffs = []
    # List of guard numbers (same index as cutoffs)
    guards_n = []
    lost_probs = []

    # Calculate number of guards for different speed thresholds
    for cutoff in xrange(0, THRESHOLD_MAX, THRESHOLD_STEP):
        # Prune the guard list (remove slow guards)
        guards.prune_guards(cutoff)
        cutoffs.append(cutoff)
        lost_probs.append(1 - sum(guards.get_prob_distr()))

    return lost_probs, cutoffs

def main():
    # Print usage if corrupted CLI
    if len(sys.argv) < 3:
        print "usage: %s <consensus file> <descriptor file>" % sys.argv[0]
        print "example: %s cached-consensus cached-descriptors" % sys.argv[0]
        sys.exit(1)

    consensus_fname = sys.argv[1]
    descriptors_fname = sys.argv[2]

    # Initialize matplotlib/seaborn/etc.
    init_graph_style()

    guards = util.parse_consensus(consensus_fname)
    guards.update_guard_attributes()
    consensus_guards_n, consensus_cutoffs = do_analysis(guards)

    guards = util.parse_consensus(consensus_fname, descriptors_fname)
    guards.update_guard_attributes()
    desc_guards_n, desc_cutoffs = do_analysis(guards)

    assert(consensus_cutoffs == desc_cutoffs)

    # Change frequency of x-axis ticks
    fig, ax = plt.subplots()
    ax.set_xticks(xrange(0, THRESHOLD_MAX, 500))
    ax.set_yticks((0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    ax.set_title("Probabilities of discarded guards being selected")
    plt.xlabel("Bandwidth cutoff in kB/s")
    plt.ylabel("Probability of discarded guards being selected")
    plt.ylim(0,1)

    speed = pd.Series(["descriptor bw", "consensus bw"])
    graphs = np.dstack((desc_guards_n, consensus_guards_n))

    sns.tsplot(graphs, condition=speed, time=desc_cutoffs)
    plt.show()

if __name__ == '__main__':
#    logging.getLogger("").setLevel(logging.DEBUG)
    main()

