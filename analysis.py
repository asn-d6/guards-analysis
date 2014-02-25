"""This software tries to understand if it's a good idea to reduce the
number of guards to 1 given the current state of the Tor network.

For now, it calculates the diversity of the current guard selection
process [0]. Then it removes slow guards from the consensus (based on
a user specified threshold), recalculates the guard bandwidth weights,
and reevaluates the diversity of the guard selection process.

[0]: we use the shannon entropy for this (like we did in#6232)

"""

import sys
import os
import logging
import math

# Import guard utilities
import util

def analysis(guards, speed_threshold):
    # Consensus parsing is done: calculate all the guard attributes.
    guards.update_guard_attributes()

    # Calculate the current diversity of the guard selection process.
    original_diversity = guards.get_diversity()

    # Prune the guard list (remove slow guards)
    guards.prune_guards(speed_threshold)

    # Pruning is done: recalculate the guard attributes.
    guards.update_guard_attributes()

    # Recalculate diversity based on the pruned guard list
    pruned_diversity = guards.get_diversity()

    logging.warning("Original: %s. Pruned: %s", original_diversity, pruned_diversity)

def main():
    # Print usage if corrupted CLI
    if len(sys.argv) < 3:
        print "usage: %s <consensus file> <speed threshold in kB>" % sys.argv[0]
        print "example: %s cached-consensus 200" % sys.argv[0]
        sys.exit(1)

    # Get speed threshold and consensus filename from CLI
    speed_threshold = int(sys.argv[2])
    consensus_fname = sys.argv[1]

    # Parse the consensus
    guards = util.parse_consensus(consensus_fname)

    # Do the analysis
    analysis(guards, speed_threshold)

if __name__ == '__main__':
#    logging.getLogger("").setLevel(logging.DEBUG)
    main()
