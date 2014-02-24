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

# A set of guards
class Guards(object):
    def __init__(self):
        self.guards = []

        self.Wgd = None
        self.Wgg = None

    # Set the Wgd and Wgg consensus parameters.
    def set_consensus_parameters(self, Wgd, Wgg):
        self.Wgd = Wgd
        self.Wgg = Wgg

    # Add a guard to our list.
    def add(self, guard):
        self.guards.append(guard)

    # Update guard-related stats based on the current state of this
    # class. That is, it recalculates guard bandwidth weights, guard
    # probabilities, etc.

    # This function also influences the individual guard objects (to
    # update their probs).
    def update_guard_attributes(self):
        assert(self.Wgd)
        assert(self.Wgg)

        total_guard_weight = 0.0

        # Calculate guard bandwidth weights
        for guard in self.guards:
            if not guard.bandwidth:
                logging.warning("guard without bandwidth!")
                continue

            # Calculate guard weight based on its flags
            if guard.is_also_exit: # this node is guard && exit
                guard_weight = self.Wgd*guard.bandwidth
            else: # this node is just a guard
                guard_weight = self.Wgg*guard.bandwidth

            # Update total guard weight
            total_guard_weight += guard_weight

            # Set the guard weight for each guard
            guard.set_guard_weight(guard_weight)

        logging.debug("[*] total_guard_weight: %s", total_guard_weight)

        # Calculate guard probabilities
        for guard in self.guards:
            guard_prob = guard.guard_weight/total_guard_weight
            # Set the guard probability for each guard
            guard.set_guard_prob(guard_prob)
            logging.debug("Guard prob: %s", guard_prob)

        # Correctness: Make sure that all probs add up to 1
        sum_guard_prob = 0.0
        for guard in self.guards:
            sum_guard_prob += guard.guard_prob
        assert(int(sum_guard_prob*100) == int(100))

    # Get the diversity of the current guard selection
    def get_diversity(self):
        # For now just return the entropy of the random variable of
        # guard probabilities.
        entropy = 0.0
        for guard in self.guards:
            prob = guard.guard_prob
            assert(prob)

            entropy += -(prob * math.log(prob, 2))

        # max entropy of a uniform prob distr is the log of its size
        max_entropy = math.log(len(self.guards), 2)

        logging.warning("Entropy: %s (max entropy of %d guards: %s).",
                        entropy, len(self.guards), max_entropy)
        return entropy


    # Prune slow guards. Compare user-provided threshold with the
    # bandwidth value provided in the "w Bandwidth=" line.
    def prune_guards(self, threshold):
        new_guard_list = []

        # Foreach guard, if we are above the speed threshold, keep the
        # guard.  Otherwise, remove it from the list.
        for guard in self.guards:
            if guard.bandwidth >= threshold:
                new_guard_list.append(guard)

        logging.warning("Before pruning: %d guards. After pruning: %d guards",
                        len(self.guards), len(new_guard_list))

        self.guards = new_guard_list

    def __len__(self):
        return len(self.guards)

# A guard
class Guard():
    def __init__(self):
        # bandwidth in kilobytes per second according to dir-spec.txt
        self.bandwidth = None
        self.is_also_exit = None

        self.guard_weight = None
        self.guard_prob = None

    def set_bandwidth(self, bw):
           self.bandwidth = bw

    def set_flags(self, values):
           if "Exit" in values and not "BadExit" in values:
               self.is_also_exit = True

    def set_guard_weight(self, guard_weight):
        self.guard_weight = guard_weight

    # Set the probability that this node will be selected as a guard
    def set_guard_prob(self, guard_prob):
        self.guard_prob = guard_prob

# XXX is this correct?
def parse_bw_weights(values):
    data = {}
    try:
        for value in values:
            key, value = value.split("=")
            data[key] = float(value) / 10000
        return data
    except:
        return None

# Parse consensus. If a relay is a guard, parse its attributes,
# and add it to our guard list.
def parse_consensus(fname):
    guards = Guards()

    with open(fname, 'r') as f:
        guard = None
        Wgd, Wgg = 1, 1

        for line in f.readlines():
            key = line.split()[0]
            values = line.split()[1:]
            if key =='r': # new router line
                guard = None

            elif key == 's': # flags line
                if not "Guard" in values: # ignore non-guards
                    continue

                # It's a guard. Instantiate a new guard!
                guard = Guard()
                guard.set_flags(values)
                guards.add(guard)

            elif key == 'w': # bandwidth line
                if not guard:
                    continue

                # It's the bandwidth details of a guard!
                # XXX is this the bandwidth value we should use?
                # XXX is this kB/s or unitless???
                router_bw = int(values[0].split('=')[1])
                guard.set_bandwidth(router_bw)

            elif key == 'bandwidth-weights': # bandwidth weights line
                data = parse_bw_weights(values)
                try:
                    Wgd = data['Wgd']
                    Wgg = data['Wgg']
                except:
                    logging.warning("Weight not found!")
                    pass

                guards.set_consensus_parameters(Wgd, Wgg)
                logging.debug("[*] Wgd: %s. Wgg: %s", Wgd, Wgg)

    assert(len(guards) > 0)

    return guards

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
    guards = parse_consensus(consensus_fname)

    # Do the analysis
    analysis(guards, speed_threshold)

if __name__ == '__main__':
#    logging.getLogger("").setLevel(logging.DEBUG)
    main()
