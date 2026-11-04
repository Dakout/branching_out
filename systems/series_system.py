import yaml
import numpy as np

from components.component import Component
from systems.basic_system import BasicSystem
import my_utils as mu


# system failure does not reset all states
class SeriesSystem(BasicSystem):
    def __init__(
        self,
        sys_cf: dict | None = None,
        sys_cf_path: None | str = "./systems/system_config.yaml",
        comps: list[Component] | None = None,
        comp_cfs: list[dict] | None = None,
        reset_flag: bool = True,
    ) -> None:
        """
        Each component is a separate simulator
        * initialize each component separately with a separate config (if no list passed, just use the same default one)
        * maybe load failure times of components i

        have a system state matrix of size (B,M,H), where
            * B: number of parallel systems
            * M: number of components in the system
            * H: simulated horizon
        pre-initialize with nan the whole matrix. Since the a failure does not reset all components,
        the observations cannot be precomputed (since the first inspection depends on the previous failure)
        once a component fails -> get new Tf -> compute muks -> insert at relevant position into nan vector
        this way, we can step through time in parallel
        """
        if sys_cf is None:
            if sys_cf_path is None:
                raise RuntimeError(
                    """Must provide a default sys_cf_path to the system
                    if no config dict is passed!"""
                )
            else:
                with open(sys_cf_path, "r") as file:
                    sys_cf = yaml.load(file, Loader=yaml.FullLoader)
        super().__init__(sys_cf=sys_cf, comps=comps, comp_cfs=comp_cfs)

        if reset_flag:
            self.reset()

        self.get_all_comp_states()
        self.comp_resets = np.zeros(shape=(self.B, self.M), dtype=bool)
        return

    def get_all_comp_states(self) -> None:
        """
        Computes all possible component vector states and stores them as (1, 2**M, M)
        """
        dtype = mu.get_necessary_dtype(self.M)
        states = np.arange(2**self.M, dtype=dtype)
        shifts = np.arange(self.M - 1, -1, -1, dtype=dtype)

        self.all_states = ((states[:, None] >> shifts) & 1).astype(np.uint8)[None, ...]
        return

    def reset(self):
        return self._reset()

    # return a cost vector and state vector associated with taking an action vector
    def step(self, a: np.ndarray) -> list:
        """
        Args with number of components (M), trajectories (B)
            a (B, M): action array

        Returns:
            s (B, M): next observation muks array
            c (B, M): new cost array
            done: flag for termination of rollout
            info: dict with information

        Is it not checked that the replaced components potentially fail before
        even the next first inspection. It is assumed that this does not happend
        -> adjust T_F-distribution such that this very likely does not happen
        """

        all_tfs = np.vstack([m.tfs for m in self.comps]).T
        all_ruls = np.vstack([m.ruls for m in self.comps]).T
        reset_flag = np.zeros(shape=(self.B, self.M), dtype=bool)
        res_mat = np.zeros(shape=(self.B, self.M))
        c = np.zeros(shape=(self.B, self.M))
        t = np.zeros_like(c)

        # check for actions
        a_mask = np.where(a)
        if a_mask[0].size:
            # divide campaign equally over all components
            b_ind, m_counts = np.unique(a_mask[0], return_counts=True)
            camp_vec = np.zeros(self.B)
            camp_vec[b_ind] = self.cf["c_f"] / m_counts
            # pay variable costs + partial campaign costs
            c[a_mask] = (
                np.ones((self.B, self.M)) * self.c_vs + np.expand_dims(camp_vec, axis=1)
            )[a_mask]
            # tlcs
            t[a_mask] = all_tfs[a_mask] - all_ruls[a_mask]
            # reset repaired components
            reset_flag[a_mask] = 1
            # residual time until next inspection is simply the step size
            res_mat[a_mask] = self.cf["Delta_T"]

        not_a_mask = a == 0
        fail = all_ruls < self.cf["Delta_T"]
        if fail.any() and not_a_mask.any():
            f_mask = np.where(np.logical_and(fail, not_a_mask))
            if f_mask[0].size:
                c[f_mask] = (np.ones((self.B, self.M)) * self.c_cs)[f_mask]
                t[f_mask] = all_tfs[f_mask]
                reset_flag[f_mask] = 1
                res_mat[f_mask] = self.cf["Delta_T"] - all_ruls[f_mask]
        else:
            f_mask = (np.array([], dtype=np.int64),) * 2

        # output checking
        self.checks(res_mat=res_mat, a_mask=a_mask, f_mask=f_mask, a=a)

        # stream the costs to the total list and to individual components
        self.stream_costs(
            a_mask=a_mask,
            f_mask=f_mask,
            c=c,
            t=t,
            reset_flag=reset_flag,
        )

        # replace preventively / correctively replaced components
        self.comp_resets = reset_flag
        for i, m in enumerate(self.comps):
            # reset all repaired/failed components
            if reset_flag[:, i].any():
                m._replace(res_vec=res_mat[:, i])

        # check termination due to end of trajectory or out of samples
        out_of_samples = any(m.done for m in self.comps)
        horizon_reached = self.step_counter >= self.cf["H"]
        self.done = out_of_samples or horizon_reached

        ## now advance to next decision time
        if not self.done:
            # non-replaced comonents also deduct Delta_T
            res_mat[res_mat == 0] = self.cf["Delta_T"]
            for i, m in enumerate(self.comps):
                m.ruls -= res_mat[:, i]
                m.index += 1

            self.step_counter += 1
            if self.step_counter % 50 == 0:
                print(f"Step: {self.step_counter}")

            s = self._get_muks()

        else:
            s = None

        info = {}
        return s, c, self.done, info

    def stream_costs(
        self,
        a_mask: tuple,
        f_mask: tuple,
        c: np.ndarray,
        t: np.ndarray,
        reset_flag: np.ndarray,
    ) -> None:
        """
        Logging of all occurrences into self.total_costs and self.total_tlcs
        """
        r_rows, r_cols = a_mask
        if r_rows.size:
            r_costs = c[a_mask]
            r_tlcs = t[a_mask]
            self.log_events(b_inds=r_rows, m_inds=r_cols, costs=r_costs, tlcs=r_tlcs)
            # replacement
        f_rows, f_cols = f_mask
        if f_rows.size:
            f_costs = c[f_mask]
            f_tlcs = t[f_mask]
            self.log_events(b_inds=f_rows, m_inds=f_cols, costs=f_costs, tlcs=f_tlcs)
        # stream into individual components (for check)
        for i, m in enumerate(self.comps):
            if c[:, i].any():
                m.costs += c[:, i]
                m.tlcs += t[:, i]
                m.counts += reset_flag[:, i]
        return

    def checks(
        self, res_mat: np.ndarray, a_mask: tuple, f_mask: tuple, a: np.ndarray
    ) -> None:
        """
        Performs some checks to detect potential bugs in the implementation.
        """
        assert (res_mat >= 0).all()
        # if a_mask or f_mask are empty -> they cannot contain a common index
        if not a_mask[0].size:
            overlap_check = False
        elif not f_mask[0].size:
            overlap_check = False
        else:
            overlap_check = all(
                [np.array_equal(a1=a_mask[k], a2=f_mask[k]) for k in range(len(a_mask))]
            )
        assert not overlap_check
        return

    def sys_rel(self, inds: np.ndarray, muks: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """
        Returns the series system's reliability.
        Either the full batch of paralell envs passed in here (B) and the reliability is calculated
        in a vectorized manner for all of them. Or the reliability is computed for all possible system
        states but only for a single trajectory.
        -> X could be either B or 2^(M)-1
        -> Y could be either B or 1

        Args:
            inds (X, M): component flags - True stands for a replaced component (pf=0)
            muks (Y, M): means of component ln RUL-PDFs
            ub (Y, 1): upper time bound for the reliability calculation

        Returns:
            res (X, 1): series system reliability R = \prod_i R_i = \prod_i (1-F_i)
        """
        res = np.ones_like(ub, dtype=float)
        for i, m in enumerate(self.comps):
            res *= 1 - (1 - inds[:, i]) * m.RUL_pred_k_cdf(x=ub, muks=muks[:, i])
        return res

    def sys_CDF(self, inds: np.ndarray) -> np.ndarray:
        return 1 - self.sys_rel(inds=inds)
