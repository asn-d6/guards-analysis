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
from scipy.stats import rv_discrete

SAMPLES_N = 10**4 + 2 # how many guard samples we want for the 3 guard case

def init_graph_style():
    np.random.seed(9221999)
    sns.set(palette="Set2")

# XXX WRONG WRONG WRONG
def get_cdf_from_probs(probs):
    # Create CDF of probability / bandwidth.
    cdf = []

    for prob in probs:
        if cdf:
            prev_sum = cdf[-1]
        else:
            prev_sum = 0

        cdf.append(prev_sum + prob)

    return cdf

"""Sample SAMPLES_N guard selections, split them into groups of 3,
and consider them as the guard nodes of a user.

For each 3-tuple of guard node bandwidth, consider the max value as
the bandwidth of the user, and calculate the probability of a user
getting a specific bandwidth value

Return the bandwidth list and its probabilities.
"""
def sample_three_guards(bw_list, prob_list):
    user_bw_list = []

    # add two to make it a multiple of 3.
    logging.warning("Starting 3 guard sampling")
    values = np.array(bw_list)
    distrib = rv_discrete(values=(range(len(values)), prob_list))

    # Do 1 million samples
    samples_index = distrib.rvs(size=SAMPLES_N)
    bw_samples = values[samples_index].tolist()

    # calculate the number of users we are going to consider, assuming
    # that each user uses 3 guards
    assert(SAMPLES_N % 3 == 0)
    users_n = SAMPLES_N / 3

    # get average guard bw for each user
    for i in xrange(users_n):
        logging.debug("Choosing best from %d:%d:%d",bw_samples[3*i],bw_samples[3*i+1],bw_samples[3*i+2])
        guard_bw_1 = bw_samples[3*i]
        guard_bw_2 = bw_samples[3*i+1]
        guard_bw_3 = bw_samples[3*i+2]
        mean_guard_bw = sum((guard_bw_1, guard_bw_2, guard_bw_3))
        mean_guard_bw /= 3

        user_bw_list.append(mean_guard_bw)

    assert(len(user_bw_list) == users_n)

    # Now collapse duplicate bws and calculate probs
    orig_sample_size = len(user_bw_list)
    uniq_bw = set(user_bw_list)
    uniq_bw = sorted(uniq_bw)
    bw_probs = []

    for bw in uniq_bw:
        occurences = user_bw_list.count(bw)
        bw_prob = occurences / float(orig_sample_size)

        bw_probs.append(bw_prob)

        logging.debug("bw %d occured %d times. prob: %s",
                      bw, occurences, bw_prob)

    return uniq_bw, bw_probs

"""'basis_bw' contains all the possible bw values.  'to_normalize_bw'
contains some bw values that match to their probability space in
'to_normalize_probs'.

return a bw list and proability space that contains all the bw values
of 'basis_bw'

"""
def normalize_probs(to_normalize_bw, to_normalize_probs, basis_bw):
    normalized_probs = []

    normalized_bw = set(to_normalize_bw)
    normalized_bw = normalized_bw.union(basis_bw)
    normalized_bw = sorted(list(normalized_bw))

    for bw in normalized_bw:
        if bw in to_normalize_bw:
            bw_prob_index = to_normalize_bw.index(bw)
            bw_prob = to_normalize_probs[bw_prob_index]
            normalized_probs.append(bw_prob)
        else:
            normalized_probs.append(0)

    return normalized_bw, normalized_probs

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

    # Get the original probability distribution.
    original_prob_distr = guards.get_prob_distr()

    one_guard_bw, one_guard_probs = guards.get_probs_after_merge_duplicate_bw()
    three_guard_bw, three_guard_probs = sample_three_guards(one_guard_bw, one_guard_probs)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    if len(sys.argv) == 2:
        guards = util.parse_consensus(consensus_fname)
    elif len(sys.argv) == 3:
        guards = util.parse_consensus(consensus_fname, sys.argv[2])
    else:
        log.error("More arguments than needed. You don't know what you are doing! Exiting!")
        sys.exit(1)

    guards.prune_guards(2000)

    # Consensus parsing is done: calculate all the guard attributes.
    guards.update_guard_attributes()

    # Get the original probability distribution.
    original_prob_distr = guards.get_prob_distr()

    one_guard_bw_2000, one_guard_probs_2000 = guards.get_probs_after_merge_duplicate_bw()

    logging.debug("ORIGINAL")
    logging.debug("three bw: %s" % three_guard_bw)
    logging.debug("three probs: %s" % three_guard_probs)
    logging.debug("one bw: %s" % one_guard_bw)
    logging.debug("one probs: %s" % one_guard_probs)
    logging.debug("=========")

    three_guard_bw, three_guard_probs = normalize_probs(three_guard_bw, three_guard_probs, one_guard_bw)
    three_guard_bw, three_guard_probs = normalize_probs(three_guard_bw, three_guard_probs, one_guard_bw_2000)

    one_guard_bw, one_guard_probs = normalize_probs(one_guard_bw, one_guard_probs, three_guard_bw)
    one_guard_bw, one_guard_probs = normalize_probs(one_guard_bw, one_guard_probs, one_guard_bw_2000)

    one_guard_bw_2000, one_guard_probs_2000 = normalize_probs(one_guard_bw_2000, one_guard_probs_2000, one_guard_bw)
    one_guard_bw_2000, one_guard_probs_2000 = normalize_probs(one_guard_bw_2000, one_guard_probs_2000, three_guard_bw)

    logging.debug("NORMALIZED")
    logging.debug("three bw: %s" % three_guard_bw)
    logging.debug("three probs: %s" % three_guard_probs)
    logging.debug("one bw: %s" % one_guard_bw)
    logging.debug("one probs: %s" % one_guard_probs)
    logging.debug("=========")

    one_guard_cdf = get_cdf_from_probs(one_guard_probs)
    one_guard_cdf_2000 = get_cdf_from_probs(one_guard_probs_2000)
    three_guard_cdf = get_cdf_from_probs(three_guard_probs)

    # Change frequency of x-axis ticks
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
#    ax.set_xticks((10,1000,10000,100000))
    plt.ylim(0,1)

    if len(sys.argv) == 2:
        ax.set_title("CDF of probabilities of guard bandwidth (%d samples) (consensus bandwidth)" % SAMPLES_N)
    else:
        ax.set_title("CDF of probabilities of guard bandwidth (%d samples) (descriptor bandwidth)" % SAMPLES_N)

    plt.xlabel("Guard bandwidth in kB/s (average guard bandwidth for 3 guards)")
    plt.ylabel("CDF of guard selection")

    speed = pd.Series(["one guard", "three guards", "one guard (guard bw cutoff at 2000 kB/s)"])
    graphs = np.dstack((one_guard_cdf, three_guard_cdf, one_guard_cdf_2000))
    sns.tsplot(graphs, condition=speed, time=one_guard_bw)

    plt.show()

if __name__ == '__main__':
    #    logging.getLogger("").setLevel(logging.DEBUG)
    main()
