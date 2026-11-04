import warnings
import numpy as np

from systems.series_system import SeriesSystem

warnings.filterwarnings(
    "ignore",
    message="Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.",
)


class BasicAgent:
    def __init__(self, sys: SeriesSystem, agent_cf: dict) -> None:
        self.sys = sys
        self.cf = agent_cf
        return

    def var_R(self, c: np.ndarray, t: np.ndarray) -> float:
        """
        Computes the first-order approximation of the variance of a ratio of random
        variables.
        """
        return (
            1
            / np.size(c)
            * (
                np.var(c) / np.mean(t) ** 2
                + np.mean(c) ** 2 * np.var(t) / np.mean(t) ** 4
                - 2 * np.mean(c) * np.cov(c, t)[0, 1] / np.mean(t) ** 3
            )
        )

    def stream_checker(self):
        """
        Additional checks if the stored values at different places coincide.
        """
        for k, m in enumerate(self.sys.comps):
            for j in range(self.sys.B):
                try:
                    assert abs(m.costs[j] - np.sum(self.sys.total_costs[k][j])) < 1e-6
                    assert abs(m.tlcs[j] - np.sum(self.sys.total_tlcs[k][j])) < 1e-6
                except:
                    print(f"failed at k={k}, j={j}")
        return

    def multi_output_checker(self, opt_mode: float = None) -> list:
        """
        Args:
            opt_mode: flag whether in optimization mode

        Returns either:
            just mean performance
            list including: total mean, total variance, individual mean, individual variances
        """
        # self.stream_checker()
        if opt_mode:
            cinfs = np.array([m.costs.sum() / m.tlcs.sum() for m in self.sys.comps])
            self.sys.reset()
            return cinfs.sum()
        else:
            cinfs = []
            var_cinfs = []
            for m in range(len(self.sys.comps)):
                # collect all rewards and tlcs per component in a single flattened array
                c = np.concatenate(self.sys.total_costs[m])
                t = np.concatenate(self.sys.total_tlcs[m])
                try:
                    assert not (c == 0).any()
                    assert not (t == 0).any()
                except AssertionError:
                    print("Indices with 0 costs: ", np.where(c == 0)[0])
                    print("Indices with 0 life times: ", np.where(t == 0)[0])
                cinfs.append(np.mean(c) / np.mean(t))
                var_cinfs.append(self.var_R(c=c, t=t))

        cinfs = np.array(cinfs)
        var_cinfs = np.array(var_cinfs)
        total_cinf = cinfs.sum()
        total_var_cinf = var_cinfs.sum()
        return total_cinf, total_var_cinf, cinfs, var_cinfs
