import itertools
import numpy as np
import scipy.stats as sps

from systems.series_system import SeriesSystem


class MDP:
    """
    Average-cost MDP for a 2-component series system.

    The MDP has:
        - fine RUL state resolution given by `state_dt`
        - maintenance decisions every S.cf["Delta_T"]

    Example:
        state_dt = 1
        Delta_T = 10

    Then:
        - RUL states are 0, 1, 2, ..., R_max
        - maintenance decisions occur every 10 state steps
        - between decisions, components evolve for 10 fine-grid steps
        - maintenance is only allowed at the decision epochs

    State convention:
        r_i > 0:
            component survives at least the next fine-grid step.

        r_i = 0:
            component fails during the next fine-grid interval
            if no preventive replacement is performed.

    Important:
        A newly sampled component has RUL >= ΔT/state_dt.
        Hence a newly replaced component cannot fail before the next
        maintenance decision epoch.
    """

    def __init__(
        self,
        S: SeriesSystem,
        state_dt: float | int = 1.0,
    ):
        self.S = S

        if self.S.M != 2:
            raise RuntimeError(
                "Current implementation is only for a 2-component system."
            )

        self.state_dt = float(state_dt)

        # Maintenance interval in fine-grid state steps.
        decision_ratio = self.S.cf["Delta_T"] / self.state_dt

        if not np.isclose(
            decision_ratio,
            round(decision_ratio),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(
                f"ΔT must be an integer multiple of state_dt."
                f"Got ΔT={self.S.cf['Delta_T']}, state_dt={self.state_dt}."
            )

        self.decision_steps = int(round(decision_ratio))

        if self.decision_steps < 1:
            raise ValueError("decision_steps must be >= 1.")

        self.actions = list(itertools.product([0, 1], repeat=self.S.M))

        self._get_max_rul_steps()
        self._discretize_initial_distributions()
        return

    def _get_max_rul_steps(self) -> None:
        """
        Convert the continuous Gaussian parameters into fine-grid units.
        Simulation process excludes case of T_F<ΔT, mirror this for upper
        truncation as well, leads to max_Ri = μ_i + (μ_i-ΔT).
        `μ_i` is expressed in units of state_dt rather than units of ΔT.
        """
        self.norm_params = []
        self.max_Ri = []
        for m in self.S.comps:
            mu_phys, sigma_phys = (m.cf["mu_TF"], m.cf["sigma_TF"])
            check = mu_phys / self.state_dt
            assert np.isclose(check, int(check), atol=1e-10)
            mu = mu_phys / self.state_dt
            sigma = sigma_phys / self.state_dt
            max_R = int(2.0 * mu - self.S.cf["Delta_T"])
            self.norm_params.append((mu, sigma))
            self.max_Ri.append(max_R)
        return

    def _discretize_initial_distributions(self) -> None:
        """
        Discretize the continuous Gaussian lifetime into a fine-grid PMF.

        A newly installed component is conditioned to survive at least
        one maintenance interval: T_F >= ΔT.

        Since the fine state unit is `state_dt`, define R = floor(T_F / state_dt).
        With R >= decision_steps & p_init[0 : decision_steps] = 0.

        For r >= decision_steps:
            p_init[r] = P(r*state_dt <= T_F < (r+1)*state_dt
                    | Delta_T <= T_F < upper_truncation)

        The upper truncation is the finite state-space truncation already
        used by the previous implementation.
        """
        self.p_init = []
        D = self.decision_steps
        for i, (mu, sigma) in enumerate(self.norm_params):
            max_r = self.max_Ri[i]
            # Bin boundaries: D, D+1, ..., max_r+1
            # State r corresponds to: r <= T_F/state_dt < r+1
            edges = np.arange(D, max_r + 2, dtype=float)
            cdf_vals = sps.norm.cdf(edges, loc=mu, scale=sigma)

            normalization = cdf_vals[-1] - cdf_vals[0]
            if normalization <= 0:
                raise ValueError(f"Invalid Gaussian truncation for component {i}.")

            p = np.zeros(max_r + 1, dtype=float)
            p[D:] = np.diff(cdf_vals) / normalization

            # Numerical cleanup.
            p[:D] = 0.0
            p /= p.sum()
            self.p_init.append(p)
        return

    # ------------------------------------------------------------------
    # Preventive-maintenance cost
    # ------------------------------------------------------------------
    def get_pm_cost(self, action: np.ndarray) -> float:
        """
        Preventive replacement cost. Fixed cost is paid once whenever at
        least one component is preventively replaced.
        """
        if np.sum(action) == 0:
            return 0.0
        return self.S.cf["c_f"] + np.sum(np.asarray(action) * self.S.c_vs)

    def get_action_values(self, V: np.ndarray) -> np.ndarray:
        """
        Calculate the expected one-step cost + future relative value for
        all four actions in every state.

        Actions:
            0 -> (0, 0): replace neither
            1 -> (0, 1): replace component 2
            2 -> (1, 0): replace component 1
            3 -> (1, 1): replace both

        Args:
            V (R1+1, R2+1): Relative value function

        Returns:
            Q (R1+1, R2+1, 4): Action-value table
        """

        R1 = self.max_Ri[0]
        R2 = self.max_Ri[1]
        D = self.decision_steps

        shape = (R1 + 1, R2 + 1)

        # --------------------------------------------------------------
        # Immediate preventive-maintenance costs
        pm_cost_00 = self.get_pm_cost([0, 0])
        pm_cost_01 = self.get_pm_cost([0, 1])
        pm_cost_10 = self.get_pm_cost([1, 0])
        pm_cost_11 = self.get_pm_cost([1, 1])

        # --------------------------------------------------------------
        # Corrective-maintenance costs. With fine-grid resolution, a
        # component with r < D fails before the next decision epoch.
        c_c1 = self.S.c_cs[0]
        c_c2 = self.S.c_cs[1]

        fail_mask_1 = (np.arange(R1 + 1) < D)[:, None]
        fail_mask_2 = (np.arange(R2 + 1) < D)[None, :]

        C_CM_00 = c_c1 * fail_mask_1 + c_c2 * fail_mask_2

        # --------------------------------------------------------------
        # Distributions after PM at the current decision epoch.
        #
        # A fresh component with RUL k >= D is aged for exactly D fine
        # steps before the next decision: next_RUL = k - D.
        # Hence: fresh1_after_PM[j] = p1[j + D]
        p1 = self.p_init[0]
        p2 = self.p_init[1]

        fresh1_after_PM = p1[D:]
        fresh2_after_PM = p2[D:]

        n1_after_PM = fresh1_after_PM.size
        n2_after_PM = fresh2_after_PM.size

        # ----------------------------------------------------------
        # ACTION (0,0): replace neither
        # ==============================================================
        # Both components simply evolve according to their current
        # RULs. Components at r=0 fail and are immediately replaced.
        EV_00 = np.empty(shape, dtype=float)

        # Both components survive:
        # (r1, r2) -> (r1-D, r2-D) for r1>0, r2>0
        EV_00[D:, D:] = V[:-D, :-D]

        # Component 1 fails, component 2 survives.
        # r1 < D. Failure occurs after r1 fine-grid steps. New component
        # therefore ages: D - r1 steps before the next decision.
        for r1 in range(D):
            q1 = D - r1
            EV_00[r1, D:] = p1[q1:] @ V[: R1 - q1 + 1, :-D]

        # Component 2 fails, component 1 survives.
        # Mirror of previous case
        for r2 in range(D):
            q2 = D - r2
            EV_00[D:, r2] = V[:-D, : R2 - q2 + 1] @ p2[q2:]

        # Both fail.
        for r1 in range(D):
            q1 = D - r1
            for r2 in range(D):
                q2 = D - r2
                EV_00[r1, r2] = p1[q1:] @ V[: R1 - q1 + 1, : R2 - q2 + 1] @ p2[q2:]

        # Add CM costs for components that fail during this interval.
        Q00 = pm_cost_00 + C_CM_00 + EV_00

        # ==============================================================
        # ACTION (0,1): replace component 2
        # ==============================================================
        # Component 2 is sampled NOW and cannot fail during this interval
        # because p2[r] = 0 for r < D. Component 1 is not replaced.
        #
        # Expected future value after PM of component 2, as a function of
        # the possible next RUL of component 1.
        # E[V(r1_next, R2_next)], with R2_next distributed according to the
        # freshly replaced component 2.

        pm2_future = V[:, :n2_after_PM] @ fresh2_after_PM

        future_01 = np.empty(R1 + 1, dtype=float)

        # Component 1 survives the entire interval: current r1 >= D
        # r1 -> r1-D
        future_01[D:] = pm2_future[:-D]

        for r1 in range(D):
            q1 = D - r1
            future_01[r1] = p1[q1:] @ pm2_future[: R1 - q1 + 1]

        Q01 = pm_cost_01 + future_01[:, None] + c_c1 * fail_mask_1

        # ==============================================================
        # ACTION (1,0): replace component 1
        #
        # Symmetric to (0,1)
        # ==============================================================
        pm1_future = fresh1_after_PM @ V[:n1_after_PM, :]

        future_10 = np.empty(R2 + 1, dtype=float)
        future_10[D:] = pm1_future[:-D]

        for r2 in range(D):
            q2 = D - r2
            future_10[r2] = p2[q2:] @ pm1_future[: R2 - q2 + 1]

        Q10 = pm_cost_10 + future_10[None, :] + c_c2 * fail_mask_2

        # ==============================================================
        # ACTION (1,1): replace both
        #
        # Both new components are sampled immediately and age exactly D
        # fine-grid steps before the next decision.
        #
        # Future value is therefore independent of the current state.
        # ==============================================================
        future_11 = fresh1_after_PM @ V[:n1_after_PM, :n2_after_PM] @ fresh2_after_PM

        Q11 = pm_cost_11 + future_11

        # ----------------------------------------------------------
        # Stack action values
        #
        # 0 -> (0,0)
        # 1 -> (0,1)
        # 2 -> (1,0)
        # 3 -> (1,1)
        # ----------------------------------------------------------

        Q = np.empty(shape + (4,), dtype=float)

        Q[..., 0] = Q00
        Q[..., 1] = Q01
        Q[..., 2] = Q10
        Q[..., 3] = Q11
        return Q

    def solve_rvi(
        self,
        tol: float = 1e-5,
        max_iter: int = 1000,
    ):
        """
        Solve the average-cost MDP using Relative Value Iteration.

        Returns
        -------
        cost_rate
            Optimal long-run cost per unit time.

        best_action (R1 + 1, R2 + 1): np.ndarray
            Optimal action table with shape.
        """
        R1 = self.max_Ri[0]
        R2 = self.max_Ri[1]
        shape = (R1 + 1, R2 + 1)

        # V[s] = relative value of state s
        V = np.zeros(shape)

        ref_state = (0, 0)

        print(f"Solving RUL-MDP with {(R1 + 1) * (R2 + 1)} states using RVI.")
        for it in range(max_iter):

            # Calculate Q(s,a) for every state and action.
            Q = self.get_action_values(V)

            # Bellman minimization.
            Q_min = np.min(Q, axis=-1)
            best_action = np.argmin(Q, axis=-1)

            # Relative Value Iteration normalization.
            g = Q_min[ref_state]

            V_new = Q_min - g

            error = np.max(np.abs(V_new - V))

            V = V_new

            if error < tol:
                print(f"Converged in {it + 1} iterations.")
                cost_rate = g / self.S.cf["Delta_T"]
                return cost_rate, best_action

        print("Warning: RVI did not converge within max_iter.")
        return g / self.S.cf["Delta_T"], best_action

    def evaluate_policy(
        self,
        policy: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 1000,
    ):
        """
        Evaluate the long-run average cost rate of a fixed policy.

        Args:
            policy (R1+1, R2+1): Fixed action table
            tol: Convergence tolerance
            max_iter: maximum numer of RVI iterations

        Returns
            cost_rate: Long-run average cost per unit time
        """
        R1 = self.max_Ri[0]
        R2 = self.max_Ri[1]
        shape = (R1 + 1, R2 + 1)

        policy = np.asarray(policy, dtype=int)

        if policy.shape != shape:
            raise ValueError(f"Policy must have shape {shape}, got {policy.shape}.")

        if np.any((policy < 0) | (policy > 3)):
            raise ValueError("Policy contains invalid actions, must be [0,1,2,3].")

        V = np.zeros(shape)
        ref_state = (0, 0)

        print(f"Evaluating policy on {(R1 + 1) * (R2 + 1)} states.")
        for it in range(max_iter):
            # Exact Q-values under the current value function.
            Q = self.get_action_values(V)

            # Select the action prescribed by the fixed policy.
            Q_policy = np.take_along_axis(Q, policy[..., None], axis=-1)[..., 0]

            # Relative-value normalization.
            g = Q_policy[ref_state]

            V_new = Q_policy - g

            error = np.max(np.abs(V_new - V))

            V = V_new

            if error < tol:
                print(f"Policy evaluation converged in {it + 1} iterations.")
                return g / self.S.cf["Delta_T"]

        print("Warning: policy evaluation did not converge within max_iter.")
        return g / self.S.cf["Delta_T"]

    def heuristic_policy_on_rul_grid(
        self,
        heuristic,
        heuristics_pars: dict,
        max_rul_steps: tuple[int, int],
        rul_step: float = 1.0,
        rul_offset: float = 1e-8,
        return_muks: bool = False,
    ):
        """
        Evaluate a prognostic heuristic on every 2-component RUL combination
        and convert its component-wise actions to the same action encoding
        as the MDP.

        Args:
            heuristic: callable function, accepts (B, 2) muk-array and returns
                (B, 2) action array containing 0/1 per component.
            max_rul_steps (R1_max, R2_max): max RUL states for components 1 and 2.
            rul_step: Physical time represented by one RUL state step.
            rul_offset: Small added to RUL, e.g., to avoid ln(0)
            return_muks: flag to also return the muk values used for every state.

        Returns:
            policy (R1_max+1, R2_max+1): Action codes 0, 1, 2, 3 matching the MDP
            muks_grid (R1_max+1, R2_max+1, 2): Returned only if return_muks=True
        """

        R1_max, R2_max = max_rul_steps

        # ---------------------------------------------------------------
        # Enumerate every physical RUL combination. State (r1, r2) is:
        #
        #     rul_1 = r1 * rul_step
        #     rul_2 = r2 * rul_step
        # ---------------------------------------------------------------
        r1 = np.arange(R1_max + 1, dtype=float) * rul_step + rul_offset
        r2 = np.arange(R2_max + 1, dtype=float) * rul_step + rul_offset
        r1_grid, r2_grid = np.meshgrid(r1, r2, indexing="ij")

        # ---------------------------------------------------------------
        # Perfect-information prognostic parameters. Model is:
        #
        #     mu = log(RUL) + ln(error)
        #
        # For the deterministic perfect-information comparison:
        #
        #     ln(error) = 0 ->  mu = log(RUL)
        # --------------------------------------------------------------
        muk_grid = np.empty((R1_max + 1, R2_max + 1, 2), dtype=float)
        muk_grid[..., 0] = np.log(r1_grid)
        muk_grid[..., 1] = np.log(r2_grid)

        # Flatten state grid so heuristic can process all states simultaneously.
        muks = muk_grid.reshape(-1, 2)

        # Evaluate heuristic.
        actions = heuristic(muks, **heuristics_pars)

        actions = np.asarray(actions)

        if actions.shape != muks.shape:
            raise ValueError(
                f"Heuristic returned shape {actions.shape}, " f"expected {muks.shape}."
            )

        # ---------------------------------------------------------------
        # Convert component actions to MDP action encoding:
        #
        # (0,0) -> 0
        # (0,1) -> 1
        # (1,0) -> 2
        # (1,1) -> 3
        # ---------------------------------------------------------------

        a1 = actions[:, 0].astype(int)
        a2 = actions[:, 1].astype(int)

        if not (np.all((a1 == 0) | (a1 == 1)) and np.all((a2 == 0) | (a2 == 1))):
            raise ValueError("Heuristic actions must contain only 0/1.")

        action_codes = a1 * 2 + a2

        policy = action_codes.reshape(
            R1_max + 1,
            R2_max + 1,
        )

        if return_muks:
            return policy, muk_grid

        return policy

    def coarse_check(MO, Q):
        for k1 in range(MO.max_Ri[0] + 1):
            for k2 in range(MO.max_Ri[1] + 1):

                block = Q[
                    10 * k1 : 10 * k1 + 10,
                    10 * k2 : 10 * k2 + 10,
                ]

                if len(np.unique(block)) > 1:
                    print(
                        "Fine MDP distinguishes this coarse state:",
                        k1,
                        k2,
                        np.unique(block),
                    )
        return
