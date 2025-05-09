import math
from equilibrium_real import find_equilibrium_realistic


def best_two_crates(p, set_list, multipliers, N, container_fees=None):
    """
    Given the equilibrium distribution p[s], compute your best pair of crates.

    Steps:
    1) Compute n[i], fraction of population in crate i.
    2) For each pair (i,j),
         your share = V[i]/(n[i]*N + 1) + V[j]/(n[j]*N + 1)
         minus container fees if any.
    3) Return the best (i,j) and the resulting payoff.
    """
    import math

    if container_fees is None:
        # default to zero cost
        container_fees = [0] * len(multipliers)

    M = len(multipliers)
    V = [m * 10000 for m in multipliers]

    # 1) compute n[i]
    n = [0.0] * M
    for s_idx, s in enumerate(set_list):
        for i in s:
            n[i] += p[s_idx]

    best_pair = None
    best_payoff = -1e15

    # 2) check all pairs i<j
    for i in range(M):
        for j in range(i + 1, M):
            # your share if you join i & j
            share_i = V[i] / ((n[i] * N) + 1)
            share_j = V[j] / ((n[j] * N) + 1)
            # subtract fees
            net_payoff = share_i + share_j - (container_fees[i] + container_fees[j])
            if net_payoff > best_payoff:
                best_payoff = net_payoff
                best_pair = (i, j)

    return best_pair, best_payoff


if __name__ == "__main__":
    multipliers = [10, 37, 17, 31, 90, 50, 20, 89, 80, 73]
    N = 10000

    container_fees = [50] * len(multipliers)

    # 1) Find equilibrium
    p_final, set_list = find_equilibrium_realistic(multipliers, N=N, iterations=3000)

    # distribution (only non-trivial probabilities)
    print("Final distribution (approx equilibrium):")
    for s_idx, s in enumerate(set_list):
        if p_final[s_idx] > 0.001:
            print(f"  Set {s} -> p = {p_final[s_idx]:.4f}")

    # 2) Determine the best pair
    best_pair, best_payoff = best_two_crates(p_final,
                                                     set_list,
                                                     multipliers,
                                                     N,
                                                     container_fees)

    print("\nYour best pair:", best_pair)
    print("Best pair payoff (after fees):", f"{best_payoff:.2f}")


