import os
import copy
from deap import base, creator, tools, algorithms
from functools import partial
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
from scipy import stats as sps
from tqdm import tqdm
import yaml

import my_utils as mu
from components.component import Component
from systems.series_system import SeriesSystem as SS
from agents.replacement_agent import ReplacementAgent as RA


def eval_mult(A: RA, fun: str, par: list[np.ndarray]) -> np.ndarray:
    """
    Serialized evaluation of a heuristic.

    Args:
        A: agent to evaluate (contains failure samples already)
        fun: heuristic to evaluate
        par: vector of parameters to try the heuristic

    Returns:
        res: cost rate for each parameter entry
    """
    res = np.array(
        [
            (A.eval_rh(fun=fun, opt_mode=True, par=p) if all(~np.isnan(p)) else np.nan)
            for p in par
        ]
    ).squeeze()
    return res


def shuffle_samples(s: np.ndarray, K: int) -> list[np.ndarray]:
    """
    Shuffles the

    Args:
        s (M, N): Sample arrays for each component
        K: integer for number of shuffles for each component

    Returns:
        s_shuffled (K, B, N):
    """
    if K == 1:
        return [s]

    M, N = s.shape
    # Generate K independent random permutations for each batch
    idx = np.argsort(np.random.rand(K, M, N), axis=-1)

    # Apply the shuffles along axis=1 for each of the K versions
    s_shuffled = np.take_along_axis(s[None, :, :], idx, axis=-1)
    return list(s_shuffled)


def evaluate_ind_reshuffle(
    Ind: list, fun: str, tf_samples: np.ndarray, comp_cf: list[dict], par_dict: dict
) -> tuple:
    from components.component import Component
    from systems.series_system import SeriesSystem as SS
    from agents.replacement_agent import ReplacementAgent as RA

    """
    Given a population, this function evaluates each individual for different
    shuffled sampled instances with multiprocessing

    -> preps data and calls somehow parallel_runner
    Serialized evaluation of a heuristic.

    Args:
        Ind: creator.Individual being a list of parameters to evaluate the fitness
        fun: heuristic to evaluate
        tf_samples (M, N): failure samples for each component -> to be reshuffled
        comp_cf: configs for each component
        par_dict: dict with all the GA hyperparameter

    Returns:
        res: cost rate for each parameter entry
    """
    # shuffle data n_shuffles times
    tfs_shuffled = shuffle_samples(
        s=np.array(tf_samples), K=par_dict["n_shuffles"]
    )  # (K, M, N)
    # create comps for each shuffle
    train_comps = [
        [
            Component(
                cf=copy.deepcopy(comp_cf[m]),
                reset_flag=False,
                tf_samples=tfs_shuffled[k][m],
            )
            for m in range(len(comp_cf))
        ]
        for k in range(par_dict["n_shuffles"])
    ]
    # create sys & Agent with comps for each shuffle
    S = [
        SS(comps=train_comps[k], reset_flag=False)
        for k in range(par_dict["n_shuffles"])
    ]
    Agents = [RA(sys=S[k]) for k in range(par_dict["n_shuffles"])]
    res = np.array(
        [
            (
                Agents[i].eval_rh(fun=fun, opt_mode=True, par=Ind)
                if all(~np.isnan(Ind))
                else np.nan
            )
            for i in range(len(Agents))
        ]
    ).squeeze()
    return (np.nanmean(res),)


def register_creator() -> None:
    """
    Registers the creator once in the beginning of the process
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimize
    creator.create("Individual", list, fitness=creator.FitnessMin)
    return


def get_toolbox(n_comps: int, par_dict: dict) -> base.Toolbox:
    """
    Registers all toolbox parts and returns toolbox
    """
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=n_comps,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[0] * n_comps,
        up=[1] * n_comps,
        eta=par_dict["eta_c"],
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[0] * n_comps,
        up=[1] * n_comps,
        eta=par_dict["eta_m"],
        indpb=par_dict["p_m"],
    )
    toolbox.register("select", tools.selTournament, tournsize=par_dict["tournsize"])
    return toolbox


def run_ga(
    toolbox: base.Toolbox,
    pop: list,
    par_dict: dict,
) -> list[tools.HallOfFame, list]:
    hof = tools.HallOfFame(par_dict["n_population"], similar=np.array_equal)

    # Evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    # plot_hof(hof=hof)

    # Main Loop: Generational GA with Manual Elitism
    for gen in range(1, par_dict["n_generations"]):
        print(f"\n\n\nGeneration: {gen}\n\n\n")
        # select elites
        current_pop_sorted = tools.selBest(pop, len(pop))
        elites = [
            toolbox.clone(ind) for ind in current_pop_sorted[: par_dict["n_elites"]]
        ]

        # Select parents for the next generation via tournament selection
        offspring = toolbox.select(pop, par_dict["n_population"] - par_dict["n_elites"])
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply Crossover AND Mutation (varAnd)
        offspring = algorithms.varAnd(
            offspring, toolbox, par_dict["p_c"], par_dict["p_m"]
        )

        # Evaluate the entire new offspring population
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism step: Add elites to offspring for next generation
        offspring = offspring + elites
        pop[:] = offspring
        hof.update(pop)

    # plot_hof(hof=hof)

    # Return the Hall of Fame and the overall best individual
    return hof, hof[0]


def plot_hof(hof: tools.HallOfFame) -> None:
    x = [ind[0] for ind in hof]
    fitnesses = [ind.fitness.values[0] for ind in hof]
    # sort parameters and extract values
    plt.figure()
    plt.scatter(x, fitnesses)
    plt.show()
    return


def beta_samples(n: int, test_comps: list[Component], sys_cf: dict) -> list[object]:
    """
    Creates a beta distribution object for the system reliability of the
    probability of failure of each component
    """
    b = 4
    cr = np.array([(sys_cf["c_f"] + m.cf["c_v"]) / m.cf["c_c"] for m in test_comps])
    if n == 1:
        sys_rel = np.mean(cr)
        a = np.array([((b - 2) * sys_rel + 1) / (1 - sys_rel)])
    else:
        a = ((b - 2) * cr + 1) / (1 - cr)
    return [sps.beta(a=a[k], b=b) for k in range(a.size)]


def main(
    fun: str,
    n_train: int,
    test_comps: list[Component],
    n_trials: int,
    par: list | float | None = None,
) -> None:
    """
    Evaluates a heuristic when only a few samples per component are available.
    To optimize the parameters of rh1 and multi_level: a Gaussian Process (GP) is
    fitted and coupled with a Genetic Algorithm (GA)

    Args:
        fun: string defining function to evaluate
        n_train: number of available training samples per component
        test_comps: list of components containing test setting
        n_trials: How often to perform whole optimization + evaluation
        par: dict of heuristic-specific GA parameters

    Returns:
        None
    """
    # load tf file
    tfs = mu.load_file(f"./data/{len(test_comps)}comps/tfs_{n_train}")
    # load default system config
    with open("./systems/system_config.yaml", "r") as file:
        sys_cf = yaml.load(file, Loader=yaml.FullLoader)

    if "doa" not in fun:
        register_creator()
        if "multi_level" in fun:
            n = 1
        elif "rh1" in fun:
            n = len(test_comps)
        beta_dist = beta_samples(n=n, test_comps=test_comps, sys_cf=sys_cf)
        toolbox = get_toolbox(n_comps=n, par_dict=par)

    for j in tqdm(range(n_trials)):
        path = f"./perf/low_data/{fun}_{n_train}_{j}"
        if os.path.exists(path):
            print(f"Path {path} already exists, skipped!")
            continue
        if "doa" in fun:
            # create comps
            train_comps = [
                Component(
                    cf=copy.deepcopy(test_comps[m].cf),
                    reset_flag=False,
                    tf_samples=tfs[j][m],
                )
                for m in range(len(test_comps))
            ]
            # create sys & Agent with comps
            S = SS(sys_cf=copy.deepcopy(sys_cf), comps=train_comps, reset_flag=False)
            A = RA(sys=S)

            # get initialization parameters
            best_pars = np.array(
                [
                    m.cinf_sample_based(fun=fun, tfs=m.tf_samples, fac=par)
                    for m in A.sys.comps
                ]
            )
            hof = None

        else:
            comp_cf = [m.cf for m in test_comps]
            # use a different random shuffle for each of the evaluated points
            toolbox.register(
                "evaluate",
                partial(
                    evaluate_ind_reshuffle,
                    fun=fun,
                    tf_samples=tfs[j],
                    comp_cf=comp_cf,
                    par_dict=par,
                ),
            )

            # create the starting population
            if fun == "rh1_opt":
                # random samples according to beta distribution
                pf_init = np.array(
                    [
                        beta_dist[k].rvs(size=par["n_population"])
                        for k in range(len(test_comps))
                    ]
                ).T  # (n_pop, M)
                pop = [creator.Individual(v.tolist()) for v in pf_init]
            elif fun == "multi_level":
                # equally spaced; mirror for reliability
                sys_rel_init = 1 - beta_dist[0].rvs(size=par["n_population"])
                pop = [creator.Individual([v]) for v in sys_rel_init]
            # run the genetic algorithm
            hof, best_pars = run_ga(toolbox=toolbox, pop=pop, par_dict=par)
            # can uncomment to check visually for results
            plot_hof(hof=hof)

        # test on large dataset
        sys_cf["rng_seed"] = j
        S = SS(sys_cf=copy.deepcopy(sys_cf), comps=test_comps, reset_flag=False)
        A = RA(sys=S)
        R, var_R, cinfs, var_cinfs = A.eval_rh(fun=fun, opt_mode=False, par=best_pars)

        # intermediate saving
        tmp = {
            "cinf": cinfs,
            "var_cinf": var_cinfs,
            "R": R,
            "var_R": var_R,
            "a_cf": A.cf,
            "sys_cf": A.sys.cf,
            "comp_cf": [m.cf for m in A.sys.comps],
            "costs": A.sys.total_costs,
            "tlcs": A.sys.total_tlcs,
            "hof": hof,
            "best_pars": np.array(best_pars),
        }
        mu.save_general_file(
            store_dir="./perf/low_data",
            file_name=f"{fun}_{n_train}_{j}",
            obj_to_store=tmp,
        )
    return
