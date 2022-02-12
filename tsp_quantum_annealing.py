# quantum inspired simulated annealing for the traveling salesman problem
# From https://visualstudiomagazine.com/Articles/2022/01/20/quantum-inspired-annealing.aspx?Page=2

import numpy as np

def total_dist(route):
    d = 0.0  # total distance between cities
    n = len(route)
    for i in range(n-1):
        if route[i] < route[i+1]:
            d += (route[i+1] - route[i]) * 1.0
        else:
            d += (route[i] - route[i+1]) * 1.5
    return d

def error(route):
    n = len(route)
    d = total_dist(route)
    min_dist = n-1
    return d - min_dist

def adjacent(route, n_swaps, rnd):
    n = len(route)
    result = np.copy(route)
    for ns in range(n_swaps):
        i = rnd.randint(n)
        j = rnd.randint(n)
        tmp = result[i]
        result[i] = result[j]
        result[j] = tmp
    return result

def my_kendall_tau_dist(p1, p2):
    # p1, p2 are 0-based lists or np.arrays
    n = len(p1)
    index_of = [None] * n  # lookup into p2
    for i in range(n):
        v = p2[i]
        index_of[v] = i

    d = 0  # raw distance = number pair misorderings
    for i in range(n):
        for j in range(i+1, n):
            if index_of[p1[i]] > index_of[p1[j]]:
                d += 1
    normer = n * (n - 1) / 2.0
    nd = d / normer  # normalized distance
    return (d, nd)

def solve_qa(n_cities, rnd, max_iter, start_temperature, alpha, pct_tunnel):
    curr_temperature = start_temperature
    soln = np.arange(n_cities, dtype=np.int64)
    rnd.shuffle(soln)
    print("Initial guess: ")
    print(soln)
    print("")

    err = error(soln)
    iteration = 0
    interval = (int)(max_iter / 10)

    best_soln = np.copy(soln)
    best_err = err

    while iteration < max_iter and err > 0.0:
        # pct left determines n_swaps determines distance
        pct_iters_left = (max_iter - iteration) / (max_iter * 1.0)
        p = rnd.random()  # [0.0, 1.0]
        if p < pct_tunnel:            # tunnel
            num_swaps = (int)(pct_iters_left * n_cities)
            if num_swaps < 1:
                num_swaps = 1
        else: # no tunneling
            num_swaps = 1

        adj_route = adjacent(soln, num_swaps, rnd)
        adj_err = error(adj_route)

        if adj_err < best_err:
            best_soln = np.copy(adj_route)
            best_err = adj_err

        if adj_err < err:  # better route so accept
            soln = adj_route
            err = adj_err
        else: # adjacent is worse
            accept_p = np.exp((err - adj_err) / curr_temperature)
            p = rnd.random()
            if p < accept_p:  # accept anyway
                soln = adj_route
                err = adj_err
            # else don't accept worse route

        if iteration % interval == 0:
            (dist, nd) = my_kendall_tau_dist(soln, adj_route)
            print("iteration = %6d | " % iteration, end="")
            print("dist curr to candidate = %8.4f | "
                  % nd, end="")
            print("curr_temp = %12.4f | "
                  % curr_temperature, end="")
            print("error = %6.1f " % best_err)

        if curr_temperature < 0.00001:
            curr_temperature = 0.00001
        else:
            curr_temperature *= alpha

        iteration += 1
    return best_soln

def main():
    print("\nBegin TSP using quantum inspired annealing ")

    num_cities = 40

    print("\nSetting num_cities = %d " % num_cities)
    print("\nOptimal solution is 0, 1, 2, . . " +
          str(num_cities-1))
    print("Optimal solution has total distance = %0.1f "
          % (num_cities-1))
    rnd = np.random.RandomState(6)
    max_iter = 20_000  # 120000 finds optimal solution
    start_temperature = 100_000.0
    alpha = 0.9990
    pct_tunnel = 0.10

    print("\nQuantum inspired annealing settings: ")
    print("max_iter = %d " % max_iter)
    print("start_temperature = %0.1f "
          % start_temperature)
    print("alpha = %0.4f " % alpha)
    print("pct_tunnel = %0.2f " % pct_tunnel)

    print("\nStarting solve() ")
    soln = solve_qa(num_cities, rnd, max_iter,
                    start_temperature, alpha, pct_tunnel)
    print("Finished solve() ")

    print("\nBest solution found: ")
    print(soln)
    dist = total_dist(soln)
    print("\nTotal distance = %0.1f " % dist)

    print("\nEnd demo ")

if __name__ == "__main__":
    main()
