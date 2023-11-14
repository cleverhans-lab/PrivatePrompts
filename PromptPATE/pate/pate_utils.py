import math
import numpy as np
import sys

from privacy_utils import compute_logpr_answered
from privacy_utils import compute_logq_gnmax
from privacy_utils import compute_rdp_data_dependent_gnmax
from privacy_utils import compute_rdp_data_dependent_threshold
from privacy_utils import rdp_to_dp


def one_hot(indices, num_classes):
    """
    Convert labels into one-hot vectors.
    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.
    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    onehot = np.zeros((len(indices), num_classes))
    for i in range(len(indices)):
        onehot[i][indices[i]] = 1
    return onehot

def query(votes, threshold, sigma_threshold, sigma_gnmax, num_classes):
    """
    the votes come in with shape (num_samples, num_teachers)

    The function queries the ensemble and gives us out ALL the votes (also the ones that should not be answered).
    But it additionally gives us a list with indices of the queries that will be answered.
    We HAVE TO use this list afterwards for post-processing (to not use the queries that we did not actually answer).
    """
    num_samples = votes.shape[0]
    num_teachers = votes.shape[1]

    # Accumulate the predictions of all teachers (in a one-hot way)
    model_votes = np.zeros((num_samples,num_classes))
    for t in range(num_teachers):
        teacher_one_hot_votes = one_hot(votes[:,t], num_classes)
        model_votes += teacher_one_hot_votes


    # this is the noise that will be added to the one-hot labels
    noise_threshold = np.random.normal(0., sigma_threshold, num_samples)

    # get the vote counts for the highest class
    vote_counts = np.max(model_votes, axis=1)

    # and determine which samples we will output and answer on
    noisy_vote_counts = vote_counts + noise_threshold
    answered = noisy_vote_counts > threshold
    indices_answered = np.arange(num_samples)[answered]

    noise_gnmax = np.random.normal(0., sigma_gnmax, (num_samples, num_classes)) #todo: here dimensions might not be correct
    noisy_predictions = model_votes + noise_gnmax
    noisy_labels = np.argmax(noisy_predictions, axis=1).astype(int)


    return noisy_labels, indices_answered
    # label =
    # model_votes = one_hot(label, num_classes)
    #
    # noise_threshold = np.random.normal(0., args.sigma_threshold,
    #                                    num_samples)



def analyze_results(votes, max_num_query, dp_eps):
    print('max_num_query;', max_num_query)
    dp_eps_items = []
    # eps were added to the sum of previous epsilons - subtract the value
    # to get single epsilons.
    dp_eps_items.append(dp_eps[0])
    for i in range(1, len(dp_eps)):
        dp_eps_items.append(dp_eps[i] - dp_eps[i - 1])
    dp_eps_items = np.array(dp_eps_items)
    avg_dp_eps = np.mean(dp_eps_items)
    print('avg_dp_eps;', avg_dp_eps)
    print('min_dp_eps;', np.min(dp_eps_items))
    print('median_dp_eps;', np.median(dp_eps_items))
    print('mean_dp_eps;', np.mean(dp_eps_items))
    print('max_dp_eps;', np.max(dp_eps_items))
    print('sum_dp_eps;', np.sum(dp_eps_items))
    print('std_dp_eps;', np.std(dp_eps_items))

    # Sort votes in ascending orders.
    sorted_votes = np.sort(votes, axis=-1)
    # Subtract runner-up votes from the max number of votes.
    gaps = sorted_votes[:, -1] - sorted_votes[:, -2]

    assert np.all(gaps > 0)
    print('min gaps;', np.min(gaps))
    print('avg gaps;', np.mean(gaps))
    print('median gaps;', np.median(gaps))
    print('max gaps;', np.max(gaps))
    print('sum gaps;', np.sum(dp_eps_items))
    print('std gaps;', np.std(dp_eps_items))

    # aggregate
    unique_gaps = np.unique(np.sort(gaps))
    gap_eps = {}
    print('gap;mean_eps')
    for gap in unique_gaps:
        mean_eps = dp_eps_items[gaps == gap].mean()
        gap_eps[gap] = mean_eps
        print(f'{gap};{mean_eps}')

    return gap_eps, gaps


def analyze_multiclass_confident_gnmax(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta, file, args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the Confident GNMax mechanism.
    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        args: all args of the program
    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry
            corresponding to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the expected number of answered queries at a
            specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry
            corresponding to the order minimizing the privacy cost at a
            specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_threshold = rdp_eps_threshold_curr[idx]
        rdp_eps_gnmax = rdp_eps_total_curr[idx] - rdp_eps_threshold
        p = np.array([rdp_eps_threshold, rdp_eps_gnmax,
                      -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = len(votes)
    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)
    answered = np.zeros(n, dtype=float)
    # Current cumulative results
    rdp_eps_threshold_curr = np.zeros(len(orders))
    rdp_eps_total_curr = np.zeros(len(orders))
    answered_curr = 0
    # Iterating over all queries
    for i in range(n):
        v = votes[i]
        if sigma_threshold > 0:
            # logpr - probability that the label is answered.
            logpr = compute_logpr_answered(threshold, sigma_threshold, v)
            rdp_eps_threshold = compute_rdp_data_dependent_threshold(
                logpr, sigma_threshold, orders)
        else:
            # Do not use the Confident part of the GNMax.
            assert threshold == 0
            logpr = 0
            rdp_eps_threshold = 0

        logq = compute_logq_gnmax(v, sigma_gnmax)
        rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
            logq, sigma_gnmax, orders)
        rdp_eps_total = rdp_eps_threshold + np.exp(logpr) * rdp_eps_gnmax
        # Evaluate E[(rdp_eps_threshold + Bernoulli(pr) * rdp_eps_gnmax)^2]
        # Update current cumulative results.
        rdp_eps_threshold_curr += rdp_eps_threshold
        rdp_eps_total_curr += rdp_eps_total
        pr_answered = np.exp(logpr)
        answered_curr += pr_answered
        # Update all cumulative results.
        answered[i] = answered_curr
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break

    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    return max_num_query, dp_eps, partition, answered, order_opt




