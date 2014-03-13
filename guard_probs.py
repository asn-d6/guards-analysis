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

# Number of users picking guards at the same time
USERS_N = 500000

THRESHOLD_MAX = 10000
THRESHOLD_STEP = 100

def init_graph_style():
    sns.set_color_palette("deep", desat=.6)
    mpl.rc("figure", figsize=(8, 4))
    np.random.seed(9221999)

def main():
    # Print usage if corrupted CLI
    if len(sys.argv) < 2:
        print "usage: %s <consensus file> [<descriptor file>]" % sys.argv[0]
        print "example: %s cached-consensus cached-descriptors" % sys.argv[0]
        sys.exit(1)

    consensus_fname = sys.argv[1]

    # Initialize matplotlib/seaborn/etc.
    init_graph_style()

    if len(sys.argv) == 2:
        guards = util.parse_consensus(consensus_fname)
    elif len(sys.argv) == 3:
        guards = util.parse_consensus(consensus_fname, sys.argv[2])
    else:
        log.error("More arguments than needed. You don't know what you are doing! Exiting!")
        sys.exit(1)

    # Consensus parsing is done: calculate all the guard attributes.
    guards.update_guard_attributes()
    # List of cutoffs
    cutoffs = []
    # Number of simulatenous clients on biggest/smallest guards
    biggest_guard_clients = []
    median_guard_clients = []
    smallest_guard_clients = []

    # Calculate number of guards for different speed thresholds
    for cutoff in xrange(0, THRESHOLD_MAX, THRESHOLD_STEP):
        # Prune the guard list (remove slow guards)
        guards.prune_guards(cutoff)
        guards.update_guard_attributes()

        cutoffs.append(cutoff)

        smallest_guard = guards.get_smallest_guard()
        median_guard = guards.get_median_guard()
        biggest_guard = guards.get_biggest_guard()

        biggest_guard_clients.append(biggest_guard.guard_prob * USERS_N)
        median_guard_clients.append(median_guard.guard_prob * USERS_N)
        smallest_guard_clients.append(smallest_guard.guard_prob * USERS_N)

    # Change frequency of x-axis ticks
    fig, ax = plt.subplots()
    ax.set_xticks(xrange(0, THRESHOLD_MAX, 500))
    ax.set_title("Expected number of clients on biggest/smallest guard (consensus bw) (%d simulatenous clients)" % USERS_N)
    plt.xlabel("Bandwidth cutoff in kB/s")
    plt.ylabel("Expected number of clients")
    ax.set_yscale('log')

    sns.set(style="darkgrid", context="talk")

    speed = pd.Series(["biggest guard (%s)" % biggest_guard.nickname, "median guard", "smallest guard"])
    graphs = np.dstack((biggest_guard_clients, median_guard_clients, smallest_guard_clients))

    sns.tsplot(graphs, condition=speed, time=cutoffs)
    plt.show()


if __name__ == '__main__':
#    logging.getLogger("").setLevel(logging.DEBUG)
    main()

