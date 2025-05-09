import math


def find_equilibrium_realistic(multipliers,
                               N=10000,
                               iterations=2000,
                               step_size=0.0005,
                               tol=1e-12):
    """
    Attempt to find a stable distribution p[s] over 'pick sets' for crates,
    modeling that each crate i is split among n_i*N people if fraction n_i chooses it.

    multipliers[i]: multiplier for crate i
    N: total number of players
    iterations: max number of mass-shifting steps
    step_size: how much probability to move each iteration
    tol: convergence tolerance

    Returns
    -------
    p : list of floats
        final distribution p[s] (summing to 1),
        for s in set_list
    set_list : list of tuples
        each element is a tuple representing which crates form that set
    """

    # Build sets: all singletons + all pairs
    M = len(multipliers)
    V = [m * 10000 for m in multipliers]  # pot for each crate
    set_list = []
    # singletons
    for i in range(M):
        set_list.append((i,))
    # pairs
    for i in range(M):
        for j in range(i + 1, M):
            set_list.append((i, j))
    K = len(set_list)

    # Initialize p[s] = uniform
    p = [1.0 / K] * K

    def compute_fractions(p):
        """Compute n[i] = fraction choosing crate i."""
        n = [0.0] * M
        for s_idx, s in enumerate(set_list):
            for i in s:
                n[i] += p[s_idx]
        return n

    def compute_payoffs(p, n):
        """
        payoff[s] = sum_{i in s} [V_i / (n_i*N)] in an ideal sense;
        but we can ignore dividing by N for ranking and do payoff[s] = sum_{i in s} [V_i / n_i].
        If n_i == 0, treat that as huge payoff (since you'd basically get the entire pot alone).
        """
        payoffs = []
        for s in set_list:
            val = 0.0
            for i in s:
                if n[i] < 1e-12:
                    # nobody there => if you alone join, you'd share with ~0
                    val += V[i] / 1e-9  # artificially large
                else:
                    val += V[i] / n[i]
            payoffs.append(val)
        return payoffs

    for it in range(iterations):

        n = compute_fractions(p)
        payoffs = compute_payoffs(p, n)
        max_pay = max(payoffs)

        # Find all sets that achieve the top payoff
        best_sets = [idx for idx, val in enumerate(payoffs)
                     if abs(val - max_pay) < 1e-12]

        # Shift a small fraction of probability from lower-payoff sets to best sets
        delta = [0.0] * K
        changed = False

        for s_idx in range(K):
            if p[s_idx] > 1e-15:
                if payoffs[s_idx] + 1e-12 < max_pay:
                    amt = step_size * p[s_idx]
                    delta[s_idx] -= amt
                    for b_idx in best_sets:
                        delta[b_idx] += amt / len(best_sets)
                    changed = True

        new_p = [p[i] + delta[i] for i in range(K)]

        # re-normalize in case small rounding
        tot = sum(new_p)
        if tot < 1e-15:
            # degenerate => fallback to uniform
            new_p = [1.0 / K] * K
            tot = 1.0
        else:
            new_p = [x / tot for x in new_p]

        # check distance
        dist = sum(abs(new_p[i] - p[i]) for i in range(K))
        p = new_p

        if dist < tol and changed == False:
            break

    return p, set_list