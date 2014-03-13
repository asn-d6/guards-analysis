import logging
import math

from stem.descriptor import parse_file, DocumentHandler
from stem.descriptor.reader import DescriptorReader

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
        # XXX not entirely correct because of round() but meh
        assert(round(sum_guard_prob*100) == int(100))

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

        logging.info("Before pruning: %d guards. After pruning: %d guards",
                     len(self.guards), len(new_guard_list))

        self.guards = new_guard_list

    # Return list containing the probability distribution of the guards.
    def get_prob_distr(self):
        prob_distr = []
        for guard in self.guards:
            assert(guard.guard_prob)
            prob_distr.append(guard.guard_prob)

        return prob_distr

    def get_bw_list(self):
        bw_list = []
        for guard in self.guards:
            assert(guard.bandwidth)
            if guard.desc_bandwidth:
                bw_list.append(guard.desc_bandwidth)
            else:
                bw_list.append(guard.bandwidth)

        return bw_list

    def get_total_bw(self):
        total_bw = 0
        for guard in self.guards:
            if not guard.bandwidth:
                logging.warning("get_total_bw no bandwidth !")
                continue

            if guard.desc_bandwidth:
                total_bw += guard.desc_bandwidth
            else:
                total_bw += guard.bandwidth

        return total_bw

    def get_biggest_guard(self):
        bw_sorted_guards = sorted(self.guards, key=lambda x: x.guard_prob)
        return bw_sorted_guards[-1]

    def get_median_guard(self):
        bw_sorted_guards = sorted(self.guards, key=lambda x: x.guard_prob)
        median_n = len(bw_sorted_guards)/2
        return bw_sorted_guards[median_n]

    def get_smallest_guard(self):
        bw_sorted_guards = sorted(self.guards, key=lambda x: x.guard_prob)
        return bw_sorted_guards[0]

    def get_probs_after_merge_duplicate_bw(self):
        """Merge all guards with the same bandwidth and return the new bw list
        and probability disitribution. Useful for CDF calculation.
        """

        final_bw_list = []
        final_probs = []

        sorted_uniq_bw_list = set(self.get_bw_list())
        for bw in sorted(sorted_uniq_bw_list):
            bw_prob = 0.0

            for guard in self.guards:
                if guard.desc_bandwidth and guard.desc_bandwidth == bw:
                    bw_prob += guard.guard_prob
                elif guard.bandwidth == bw:
                    bw_prob += guard.guard_prob

            final_bw_list.append(bw)
            final_probs.append(bw_prob)

        return final_bw_list, final_probs

    def update_bw_of_guard_by_fpr(self, fingerprint, new_bw):
        for guard in self.guards:
            if guard.fingerprint == fingerprint:
                logging.debug("Updating %s from %s to %s", fingerprint, guard.bandwidth, new_bw)
                guard.set_desc_bandwidth(new_bw)


    def __len__(self):
        return len(self.guards)

# A guard
class Guard():
    def __init__(self, nickname, fingerprint):
        self.nickname = nickname
        self.fingerprint = fingerprint

        # bandwidth in kilobytes per second according to dir-spec.txt
        self.bandwidth = None
        self.desc_bandwidth = None
        self.is_also_exit = None

        self.guard_weight = None
        self.guard_prob = None

    def set_bandwidth(self, bw):
           self.bandwidth = bw

    def set_desc_bandwidth(self, bw):
           self.desc_bandwidth = bw

    def set_flags(self, values):
           if "Exit" in values and not "BadExit" in values:
               self.is_also_exit = True

    def set_guard_weight(self, guard_weight):
        self.guard_weight = guard_weight

    # Set the probability that this node will be selected as a guard
    def set_guard_prob(self, guard_prob):
        self.guard_prob = guard_prob


# Parse consensus. If a relay is a guard, parse its attributes,
# and add it to our guard list.
def parse_consensus(fname, desc_fname=None):
    guards = Guards()

    with open(fname, 'rb') as consensus_file:
        consensus = parse_file(consensus_file, 'network-status-consensus-3 1.0', document_handler = DocumentHandler.DOCUMENT).next()

        for router in consensus.routers.values():
            if 'Guard' in router.flags:
                guard = Guard(router.nickname, router.fingerprint)
                # XXX These should just be arguments to the __init__ of Guard:
                guard.set_flags(router.flags)
                guard.set_bandwidth(router.bandwidth)
                guards.add(guard)

                logging.debug("Saw %s with %s (%s)", router.nickname, router.flags, router.bandwidth)

        Wgd = consensus.bandwidth_weights['Wgd'] / float(10000)
        Wgg = consensus.bandwidth_weights['Wgg'] / float(10000)
        guards.set_consensus_parameters(Wgd, Wgg)
        logging.debug("Wgd: %s, Wgg: %s", Wgd, Wgg)

    assert(len(guards) > 0)

    # If no descriptor files were provided, we are done!
    if not desc_fname:
        return guards

    logging.warning("Using descriptor bandwidths!")
    # If a descriptor file was provided, parse it and use those bandwidths instead.
    with open(desc_fname, 'rb') as desc_file:
        descs = parse_file(desc_file, 'server-descriptor 1.0')

        # XXX most inefficient method ever.
        for desc in descs:
            bandwidth = min(desc.average_bandwidth, desc.burst_bandwidth, desc.observed_bandwidth)
            bandwidth = bandwidth >> 10 # from (bytes per second) to (kilobytes per second)
            guards.update_bw_of_guard_by_fpr(desc.fingerprint, bandwidth)

    return guards
