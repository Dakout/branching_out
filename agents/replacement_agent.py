import yaml
import numpy as np
import scipy.optimize as spo

from agents.basic_agent import BasicAgent
from systems.series_system import SeriesSystem
import my_utils as mu


class ReplacementAgent(BasicAgent):
    def __init__(
        self,
        sys: SeriesSystem,
        agent_cf: dict | None = None,
        agent_cf_path: None | str = "./agents/agent_config.yaml",
    ) -> None:
        if agent_cf is None:
            if agent_cf_path is None:
                raise RuntimeError(
                    """Must provide a default sys_cf_path to the system
                    if no config dict is passed!"""
                )
            else:
                with open(agent_cf_path, "r") as file:
                    agent_cf = yaml.load(file, Loader=yaml.FullLoader)
        super().__init__(agent_cf=agent_cf, sys=sys)
        self.get_all_actions()
        return

    def get_all_actions(self) -> None:
        """
        Computes all possible actions and stores them as (2**M, M) binary table.
        """
        M = self.sys.M
        dtype = mu.get_necessary_dtype(k=M)
        states = np.arange(2**M, dtype=dtype)
        shifts = np.arange(M - 1, -1, -1, dtype=dtype)
        self.all_acts = ((states[:, None] >> shifts) & 1).astype(np.uint8)
        return

    ##################################################################################
    ############################ Heuristic policies ##################################
    ##################################################################################

    def cvs_probs(self, pf: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the probability of each possible component vector state in the
        system table given an action.

        Args:
            pf (B, M): failure probabilities
            z (M,): action flag arrays

        Returns:
            (B, 2**M) component vector state probabilities
        """
        pf = pf[:, None, :]  # (B, 1, M)
        z = z[None, None, :]  # (1, 1, M)
        pj = (1 - z) * (
            self.sys.all_states * pf + (1 - self.sys.all_states) * (1 - pf)
        ) + z * (
            1 - self.sys.all_states
        )  # (B, 2**M, M)
        return np.prod(pj, axis=-1)  # (B, 2**M)

    def cf_cvs_table(self, cFi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the expected failure cost for each possible component vector
        state in the system table given an action.

        Args:
            cFi (B, M): expected costs in case of failure (M,):
                c_ci - c_{∞,i} * E[RUL | RUL < Delta_T]
            z (M,): action flag arrays

        Returns:
            (B, 2**M) component vector state failure costs
        """
        return np.sum(
            (1 - z[None, None, :]) * self.sys.all_states * cFi[:, None, :], axis=-1
        )

    def doa_ub_surv_all_states(self, cIi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the cost of survival by assuming replacement directly at next possible
        replacement opportunity for each possible component vector state in the system
        table given an action.

        Args:
            cI (B, M): failure probabilities (M,)   c_f + c_vi - c_{∞,i} * Delta_T
            z (B, M): action flag arrays (M,)

        Returns:
            (B, 2**M) component vector state failure costs
        """
        # individual replacement costs
        z = z[:, None, :]
        cIj_uncorr = np.sum(
            (1 - z) * (1 - self.sys.all_states) * cIi[:, None, :], axis=-1
        )
        # correction to include economic dependence
        corr = np.maximum(
            0, np.sum((1 - z) * (1 - self.sys.all_states), axis=-1, dtype=int) - 1
        )
        return cIj_uncorr - self.sys.cf["c_f"] * corr

    def doa_total_costs_given_z(
        self, cR: float, pj: np.ndarray, cFj: np.ndarray, cIj: np.ndarray | float
    ) -> np.ndarray:
        """
        Computes the total expected failure + future cost for a given action vector
        (implicit in the costs).

        Args:
            cR (1,): preventive replacement cost
            pj  (B, 2^M): probability for each possible system state
            cFj (B, 2^M): expected failure cost for each possible system state
            cIj (B, 2^M) or (2^M,) or (1,): expected survival cost for each possible system state

        Returns:
            ctot (B,):
        """
        return cR + np.sum(pj * (cFj + cIj), axis=1)

    def always_0(self, muks: np.ndarray) -> np.ndarray:
        """
        Corrective maintenance benchmark -> do nothing (0) for every input.
        """
        return np.zeros_like(muks, dtype=bool)

    def always_1(self, muks: np.ndarray) -> np.ndarray:
        """
        Replace at every timestep benchmark -> (1) for every input.
        """
        return np.ones_like(muks, dtype=bool)

    def rh1(self, muks: np.ndarray) -> np.ndarray:
        """
        First benchmark policy: component-wise failure probability thresholds.

        Args:
            muks (B, M): means of the predictive RUL-distribution

        Returns:
            a (B, M): ndarray of actions
        """
        a = np.zeros_like(muks, dtype=bool)
        for k, m in enumerate(self.sys.comps):
            a[:, k] = m.Pr_smaller_thresh(
                muks=muks[:, k],
                ub=self.sys.cf["Delta_T"],
                p_thres=(self.sys.cf["c_f"] + m.cf["c_v"]) / m.cf["c_c"],
            )
        return a

    def rh1_opt(self, muks: np.ndarray, p_thres: float | list) -> np.ndarray:
        """
        Optimized first benchmark policy.

        Args:
            muks (B, M): means of the predictive RUL-distribution
            p_thres (1,) or (M,): prescribed threshold probability

        Returns:
            a (B, M): action flags
        """
        if isinstance(p_thres, float):
            p_thres = np.repeat(p_thres, self.sys.M)

        a = np.zeros_like(muks, dtype=bool)
        for k, m in enumerate(self.sys.comps):
            a[:, k] = m.Pr_smaller_thresh(
                muks=muks[:, k],
                ub=self.sys.cf["Delta_T"],
                p_thres=p_thres[k],
            )
        return a

    def doa_ub(self, cIi: np.ndarray) -> np.ndarray:
        """
        First doa variant. Computes the survival cost as replacing each surviving
        component at the next decision opportunity t_{k+1}.

        Args:
            cIi (M,): cost estimation of a single component if it survives until t_{k+1}

        Returns:
            cIj (2^M, 2^M): cost estimations of combinations of components for all
                action and state combinations
        """
        all_acts = np.expand_dims(self.all_acts, axis=1)  # (2^M, 1, M)
        acts_states = (1 - all_acts) * (1 - self.sys.all_states)  # (2^M, 2^M, M)

        cIj_indep = np.sum(acts_states * cIi, axis=-1)  # (2^M, 2^M)
        dep = self.sys.cf["c_f"] * np.maximum(
            0, -1 + np.sum(acts_states, axis=-1, dtype=int)
        )  # (2^M, 2^M)
        cIj = cIj_indep - dep  # (2^M, 2^M)

        return cIj

    def doa_perfind(
        self, muks: np.ndarray, pfs: np.ndarray, cinfs: np.ndarray
    ) -> np.ndarray:
        """
        First doa variant. Computes the survival cost as replacing each surviving
        component right before its conditional expected RUL.

        Args:
            muks (B, M): means of the predictive RUL-distribution
            pfs (B, M): one-step probability of failures
            cinfs (M,): assumed long-running cost rate of the policy

        Returns:
            cIj (2^M, B, 2^M): cost estimations of combinations of components for all
                action and state combinations
        """
        cond_ruls_aft_dt_nn = np.full_like(muks, fill_value=np.nan)  # (B, M)
        for k, m in enumerate(self.sys.comps):
            cond_ruls_aft_dt_nn[:, k] = m.cond_RUL_bigger(
                muks=muks[:, k], ilb=self.sys.cf["Delta_T"]
            ) / (
                1 - pfs[:, k]
            )  # (B, M)

        cIi = np.expand_dims(
            self.sys.cf["c_f"] + self.sys.c_vs - cinfs * cond_ruls_aft_dt_nn, axis=1
        )  # (B, 1, M)
        cIj = np.zeros(
            (self.all_acts.shape[0], muks.shape[0], self.sys.all_states.shape[1])
        )  # (2^M, B, 2^M)
        for j, z in enumerate(self.all_acts):
            cIj[j, :, :] = np.sum((1 - z) * (1 - self.sys.all_states) * cIi, axis=-1)
        return cIj

    def partition(self, t: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """
        Computes a partition for each row of t. The partition matrix Y has the properties:
            i) reflexive: y_ii = 1
            ii) symmetric: y_ij = y_ji
            iii) transitive: y_ij = 1 & y_jk = 1 -> y_ik = 1

        Args:
            t (B, M): computed ideal replacement times

        Returns:
            Y (B, M, M): partitions
        """
        # convert to integers with specific tol
        t2 = np.round(t / tol).astype(np.int64)
        # find the groups through equal replacement times
        Y = (t2[:, :, None] == t2[:, None, :]).astype(int)
        return Y

    def doa(
        self,
        muks: np.ndarray,
        cinfs: np.ndarray,
        doa_type: str,
    ) -> np.ndarray:
        """
        Doa policy. Computes the general costs and the type-specific costs.

        Args:
            muks (B, M): means of the predictive RUL-distribution
            cinfs (M,): assumed long-running cost rate of the policy
            doa_type: "lb", "ub", "perf"

        Returns:
            a (B, M): action flags
        """
        pfs = np.full_like(muks, fill_value=np.nan)
        cond_ruls_bef_dt_nn = np.full_like(muks, fill_value=np.nan)

        for k, m in enumerate(self.sys.comps):
            pfs[:, k] = np.clip(
                m.RUL_pred_k_cdf(x=self.sys.cf["Delta_T"], muks=muks[:, k]),
                a_min=self.cf["minpf"],
                a_max=1 - self.cf["minpf"],
            )

        for k, m in enumerate(self.sys.comps):
            cond_ruls_bef_dt_nn[:, k] = (
                m.cond_RUL_smaller(muks=muks[:, k], iub=self.sys.cf["Delta_T"])
                / pfs[:, k]
            )

        ctot = np.zeros((muks.shape[0], self.all_acts.shape[0]))

        # expected cost in case of failure: c_ci - c_{∞,i} * E[RUL | RUL <= Delta_T]
        cFi = self.sys.c_cs - cinfs * cond_ruls_bef_dt_nn  # (B, M)

        # cost of preventive maintenance
        cR = (
            np.sum(self.all_acts * (self.sys.cf["c_f"] + self.sys.c_vs), axis=1)
            - np.maximum(0, self.all_acts.sum(axis=1, dtype=int) - 1)
            * self.sys.cf["c_f"]
        )  # (2^M,)

        # expected survival cost
        if doa_type == "ub":
            cIi = self.sys.cf["c_f"] + self.sys.c_vs - cinfs * self.sys.cf["Delta_T"]
            cIj = self.doa_ub(cIi=cIi)  # (2^M, 2^M)
        elif doa_type == "perfind":
            cIj = self.doa_perfind(muks=muks, pfs=pfs, cinfs=cinfs)  # (2^M, B, 2^M)
        else:
            raise NotImplementedError(
                f"""Doa surviving cost of doa_type {doa_type} not implemented !!"""
            )

        # loop over all action combinations
        for j, z in enumerate(self.all_acts):
            # probabilities of all states given action vector
            pj = self.cvs_probs(pf=pfs, z=z)  # (B, 2**M)
            # expected failure costs of all states given action vector
            cFj = self.cf_cvs_table(cFi=cFi, z=z)  # (B, 2**M)
            # compute total cost associated with an action
            ctot[:, j] = self.doa_total_costs_given_z(
                cR=cR[j], pj=pj, cFj=cFj, cIj=cIj[j]
            )  # (B,)

        ## take minimum over ctot matrix
        best_acts_int = np.argmin(ctot, axis=1)  # (B,)
        return self.all_acts[best_acts_int, :]

    def single_comb_finder(
        self,
        all_combs: np.ndarray,
        all_costs: np.ndarray,
        muk: np.ndarray,
        rel_thres: float,
        rel_dn: float,
    ) -> list[np.ndarray, float]:
        """
        Goes through all group combinations of a single system and find the one with the best
        ratio of reliability improvement to cost

        Args:
            all_combs (2^M-1, M): 0/1 array of all possible system states
            all_costs (2^M-1, 1): preventive replacement cost of each group
            muk (1, M): single system mean pars
            rel_thres (1,): global reliability threshold
            rel_dn (1,): reliability of doing nothing

        Returns:
            best_group (1, M): best group of 0/1 indicating components to be replaced
            best_imp_fac (1,): best improvement factor
        """
        best_imp_fac = -1
        best_comb = None

        # Step 1: compute the reliabilities of all possible system states
        all_rels = self.sys.sys_rel(
            inds=all_combs,
            muks=muk[None, :],
            ub=np.full(all_combs.shape[0], self.sys.cf["Delta_T"]),
        )

        # Step 2: loop over all possible group combinations
        for comb_cand, rel_cand, cost_cand in zip(all_combs, all_rels, all_costs):
            # Step 3: check if group is admissible
            if rel_cand > rel_thres:
                # Step 5: compute group improvement factor
                imp_fac_cand = (rel_cand - rel_dn) / cost_cand

                # Step 6: check if better than current best improvement factor & update
                if imp_fac_cand > best_imp_fac:
                    best_imp_fac = imp_fac_cand
                    best_comb = comb_cand

        return best_comb, best_imp_fac

    def multi_level(self, muks: np.ndarray, rel_thres: float) -> np.ndarray:
        """Multi-component replacement heuristic from the paper
        https://www.sciencedirect.com/science/article/pii/S0951832015002215

        Args:
            muks (B, M): means of the predictive RUL-distribution
            r_thres (1): system reliability threshold

        Returns:
            a (B, M): action flags
        """
        a = np.zeros_like(muks, dtype=np.uint8)

        rel_dn = self.sys.sys_rel(
            inds=np.zeros_like(muks, dtype=np.uint8),
            muks=muks,
            ub=np.full(shape=(self.sys.B,), fill_value=self.sys.cf["Delta_T"]),
        )

        inds = np.where(rel_dn < rel_thres)[0]
        if inds.size > 0:

            # Step 1: enumerate all potential group combinations in binary (2^M - 1)
            all_combs = self.sys.all_states[0, 1:, :]  # (2^M - 1, M)

            # Step 2: calculate the costs of all combinations
            all_costs = (
                np.sum(all_combs * self.sys.c_vs, axis=1, keepdims=True)
                + np.ones((all_combs.shape[0], 1)) * self.sys.cf["c_f"]
            )  # (2^M - 1, 1)

            # Step 3: find the combination with the best reliability to cost ratio
            for ind in inds:
                best_comb, best_imp_fac = self.single_comb_finder(
                    all_combs=all_combs,
                    all_costs=all_costs,
                    muk=muks[ind, :],
                    rel_thres=rel_thres,
                    rel_dn=rel_dn[ind],
                )
                if best_comb is not None:
                    assert best_imp_fac > 0
                    a[ind, :] = best_comb
        return a

    def eval_rh(
        self,
        fun: str,
        opt_mode: bool = False,
        par: list | float | None = None,
    ) -> list:
        """
        Evalulation of a heuristic with the initialized configs.

        Args:
            fun: heuristic to evaluate
            opt_mode: boolean flag for optimization handling. Decides whether to return
                all values or only R from the output_checker
            par: potential parameter for some heuristics

        Returns either:
            a) [R_tot, var_R_tot, Ri, var_Ri]
            b) [R_tot]
            depending on the opt_mode flag
        """
        muks = self.sys._get_muks()

        if "doa" in fun:
            doa_type = fun.split("_")[1]
            # if 'par' is a list/array with entries for each component -> assume par ist cinf
            if isinstance(par, list) or isinstance(par, np.ndarray):
                assert len(par) == len(self.sys.comps)
                cinfs = np.array(par)
            # otherwise assume par is for interpolation between init opt1 and opt2
            else:
                cinfs = []
                for m in self.sys.comps:
                    if m.tf_samples is not None:
                        cinfs.append(
                            m.cinf_sample_based(fun=fun, tfs=m.tf_samples, fac=par)
                        )
                    else:
                        cinfs.append(m.cinf(fun=fun, fac=par))
                cinfs = np.array(cinfs)

        done = False
        while not done:
            # Step 1. get actions
            if fun == "rh1":
                actions = self.rh1(muks=muks)
            elif fun == "rh1_opt":
                actions = self.rh1_opt(muks=muks, p_thres=par)
            elif "doa" in fun:
                actions = self.doa(muks=muks, cinfs=cinfs, doa_type=doa_type)
            elif "multi_level" in fun:
                actions = self.multi_level(muks=muks, rel_thres=par)
            elif fun == "always_0":
                actions = self.always_0(muks=muks)
            elif fun == "always_1":
                actions = self.always_1(muks=muks)
            else:
                raise Exception(
                    f"Function {fun} not found in list of provided functions!"
                )
            # Step 2. make environment step
            muks, costs, done, info = self.sys.step(a=actions)

        return self.multi_output_checker(opt_mode=opt_mode)

    def optimize(self, fun: str, mode: str = "multi") -> object:
        """
        Optimization function, depends on the
        """

        x0 = [(self.sys.cf["c_f"] + m.cf["c_v"]) / m.cf["c_c"] for m in self.sys.comps]
        bounds = (0, 1)

        if mode == "multi":
            if fun == "multi_level":
                x0 = 1 - np.mean(x0)

            bounds = [bounds] * self.sys.M

            res = spo.minimize(
                fun=lambda p: self.eval_rh(
                    fun=fun,
                    opt_mode=True,
                    par=p,
                ),
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.cf["maxiter"],
                    "disp": 3,
                    "eps": self.cf["bfgs_eps"],
                },
            )
        else:
            res = spo.minimize_scalar(
                fun=lambda p: self.eval_rh(
                    fun=fun,
                    opt_mode=True,
                    par=p,
                ),
                bounds=bounds,
                tol=self.cf["oacca"],
                options={
                    "maxiter": self.cf["maxiter"],
                    "disp": 3,
                },
            )
        return res
