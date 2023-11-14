import numpy as np
import scipy.stats
import math


def _logsumexp(x):
    """
    Sum in the log space.
    An addition operation in the standard linear-scale becomes the
    LSE (log-sum-exp) in log-scale.
    Args:
        x: array-like.
    Returns:
        A scalar.
    """
    x = np.array(x)
    m = max(x)  # for numerical stability
    return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
    """
    Numerically stable computation of log(1-exp(x)).
    Args:
        x: a scalar.
    Returns:
        A scalar.
    """
    assert x <= 0, "Argument must be positive!"
    # assert x < 0, "Argument must be non-negative!"
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    else:
        return -np.inf

###############################
# RDP FOR THE GNMAX MECHANISM #
###############################
def compute_logq_gnmax(votes, sigma):
    """
    Computes an upper bound on log(Pr[outcome != argmax]) for the GNMax
    mechanism.
    Implementation of Proposition 7 from PATE 2018 paper.
    Args:
        votes: a 1-D numpy array of raw ensemble votes for a given query.
        sigma: std of the Gaussian noise in the GNMax mechanism.
    Returns:
        A scalar upper bound on log(Pr[outcome != argmax]) where log denotes natural logarithm.
    """
    num_classes = len(votes)
    variance = sigma ** 2
    idx_max = np.argmax(votes)
    votes_gap = votes[idx_max] - votes
    votes_gap = votes_gap[np.arange(num_classes) != idx_max]  # exclude argmax
    # Upper bound log(q) via a union bound rather than a more precise
    # calculation.
    logq = _logsumexp(
        scipy.stats.norm.logsf(votes_gap, scale=math.sqrt(2 * variance)))
    return min(logq,
               math.log(1 - (1 / num_classes)))  # another obvious upper bound


def compute_logpr_answered(threshold, sigma_threshold, votes):
    """
    Computes log(Pr[answered]) for the threshold mechanism.
    Args:
        threshold: the threshold (a scalar).
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        votes: a 1-D numpy array of raw ensemble votes for a given query.
    Returns:
        The value of log(Pr[answered]) where log denotes natural logarithm.
    """
    return scipy.stats.norm.logsf(threshold - round(max(votes)),
                                  scale=sigma_threshold)


def compute_rdp_data_dependent_gnmax(logq, sigma, orders):
    """
    Computes data-dependent RDP guarantees for the GNMax mechanism.
    This is the bound D_\lambda(M(D) || M(D'))  from Theorem 6 (equation 2),
    PATE 2018 (Appendix A).
    Bounds RDP from above of GNMax given an upper bound on q.
    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax
            mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.
    Returns:
        A numpy array of upper bounds on RDP for all orders.
    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    # See Appendix A in PATE 2018.
    # rdp_order2 = sigma * math.sqrt(-logq)
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    # The GNMAx mechanism satisfies:
    # (order = \lambda, eps = \lambda / sigma^2)-RDP.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(
                1 + 1 / (rdp_order2 - 1)))):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
                log1mq - _log1mexp((logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)

        # Values of q close to 1 could result in a looser bound, so minimum
        # between the data dependent bound and the data independent bound
        # rdp_esp = orders / variance is taken.
        rdp_eps[mask] = np.minimum(rdp_eps, log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps

def compute_rdp_data_dependent_threshold(logpr, sigma, orders):
    """
    Computes data-dependent RDP guarantees for the threshold mechanism.
    Args:
        logpr: the value of log(Pr[answered]) for the threshold mechanism.
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.
    Returns:
        A numpy array of upper bounds on RDP for all orders.
    Raises:
        ValueError: if the inputs are invalid.
    """
    logq = min(logpr, _log1mexp(logpr))
    # The input to the threshold mechanism has sensitivity 1 rather than 2 as
    # compared to the GNMax mechanism, hence the sqrt(2) factor below.
    return compute_rdp_data_dependent_gnmax(logq, 2 ** .5 * sigma, orders)


def rdp_to_dp(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]

