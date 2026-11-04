import numpy as np
from components.component import Component


class BasicSystem:
    def __init__(
        self,
        sys_cf: dict,
        comps: list[Component] | None = None,
        comp_cfs: list[dict] | None = None,
        reset_flag: bool = True,
    ) -> None:
        self.cf = sys_cf

        if comps is None:
            self._init_components(comp_cfs=comp_cfs)
        else:
            self.comps = comps
        self._check_comps()
        np.random.seed(self.cf["rng_seed"])

        if any([m.tf_samples is not None for m in self.comps]):
            self.cf["H"] = np.inf
        self.M = len(self.comps)
        self.H = self.cf["H"]
        self.B = self.comps[0].cf["B"]

        if reset_flag:
            self._reset()
        return

    def _init_components(self, comp_cfs: list[dict] | dict | None = None) -> None:
        if comp_cfs is None:
            comp_cfs = [None] * self.cf["n_comps"]
            print("Components all initialized with default config!")
        self.comps = [Component(cf=cf, reset_flag=False) for cf in comp_cfs]
        return

    def _check_comps(self) -> None:
        for m in self.comps:
            m.cf["Delta_T"] = self.cf["Delta_T"]
            m.cf["H"] = self.cf["H"]
            m.cf["c_f"] = self.cf["c_f"]
            assert m.cf["B"] == self.comps[0].cf["B"]
        self.c_vs = np.array([m.cf["c_v"] for m in self.comps])
        self.c_cs = np.array([m.cf["c_c"] for m in self.comps])
        return

    def _get_muks(self) -> np.ndarray:
        return np.vstack([m.muks[:, m.index] for m in self.comps]).T

    def _reset(self) -> None:
        self.counts = np.zeros((self.B, self.M))  # counter per comp
        self.costs = np.zeros_like((self.B, self.M))  # cumulative cost per comp
        self.tlcs = np.zeros_like((self.B, self.M))  # cumulative tlcs per comp

        # record all costs and lifetime occurrences per comp
        self.total_costs = [[[] for _ in range(self.B)] for _ in range(self.M)]
        self.total_tlcs = [[[] for _ in range(self.B)] for _ in range(self.M)]

        self.step_counter = 0
        self.done = False

        [m._reset() for m in self.comps]
        return self._get_muks(), None, None, None

    # loops over all trajectories & components and logs the costs & lifetimes
    def log_events(
        self,
        b_inds: np.ndarray,
        m_inds: np.ndarray,
        costs: np.ndarray,
        tlcs: np.ndarray,
    ) -> None:
        for b, m, c, t in zip(b_inds, m_inds, costs, tlcs):
            self.total_costs[m][b].append(c)
            self.total_tlcs[m][b].append(t)
        return
