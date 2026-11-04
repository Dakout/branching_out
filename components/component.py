import yaml
import numpy as np

from components.simulator import Simulator


class Component(Simulator):
    def __init__(
        self,
        cf: dict | None = None,
        cf_path: None | str = "./components/component_config.yaml",
        reset_flag: bool = True,
        tf_samples: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            cf: dict with all config parameters
            cf_path: path to a yaml file containing a config dict
            reset_flag: boolean flag whether to reset directly in init or not
            tf_samples (B, N): array of failure samples -> for overfit evaluation
        """
        if cf is None:
            if cf_path is None:
                raise RuntimeError(
                    """Must provide a default cf_path to the component
                    if no config dict is passed!"""
                )
            else:
                with open(cf_path, "r") as file:
                    cf = yaml.load(file, Loader=yaml.FullLoader)
        super().__init__(cf)

        self.tf_samples = tf_samples
        if self.tf_samples is not None:
            self.tf_idx = -1
            self.cf["B"] = 1
        self.B = self.cf["B"]

        if reset_flag:
            self._reset()
        return

    ## implement here additional functionality (maybe compatibility with Gym?)
    def _draw_new(self, inds: np.ndarray, res_vec: np.ndarray | None = None):
        """
        Args:
            inds (B,): boolean array indicating whether to replace sample or not
            res_vec (B,): residual time array until the next inspection interval

        Returns:
            tf_vec (B,): failure time array
            muks (B, X): means of predicted RUL parameters, where X=⌊(max(tf)-ΔT)/ΔT⌋
        """
        if self.tf_samples is None:
            # draw new tf samples from simulator
            tf_vec = self.draw_tf_samples_bigger_val(
                Ns=inds.size, val=self.cf["Delta_T"]
            )
        else:
            # get next tf sample from deterministic samples
            self.tf_idx += 1
            if self.tf_idx >= self.tf_samples.size:
                # end of samples reached -> abort evaluation
                return None, None
            tf_vec = np.array([self.tf_samples[self.tf_idx]])
        # get decision times depening on res_time
        muks = self.muks_given_tf_vec(tf_vec=tf_vec, res_vec=res_vec)
        return tf_vec, muks

    def _reset(self) -> None:
        """
        Initializes all the needed vectors, and draws new failure time and RUL-PDF parameter samples.
        """
        self.costs = np.zeros(self.B)  # cumulative cost per comp
        self.tlcs = np.zeros(self.B)  # cumulative tlcs per comp
        res_vec = np.full((self.B,), fill_value=self.cf["Delta_T"], dtype=int)
        if self.tf_samples is not None:
            self.tf_idx = -1
        self.tfs, self.muks = self._draw_new(
            inds=np.ones(self.B, dtype=int), res_vec=res_vec
        )
        self.ruls = self.tfs - self.cf["Delta_T"]
        self.counts = np.zeros_like(self.ruls, dtype=np.int16)
        self.index = 0
        self.done = False
        return

    def _replace(self, res_vec: np.ndarray) -> None:
        """
        Replaces all failed/pr components.

        Args:
            res_vec (B,): array of residual times until next decision interval from time of failure/pr.
        """
        inds = np.where(res_vec)[0]
        # first get new batch of tfs and muks
        new_tfs, new_muks = self._draw_new(inds=inds, res_vec=res_vec[inds])
        if new_tfs is None:
            self.done = True
            return

        # check if new batch or existing one is smaller -> pad smaller
        n_cols = new_muks.shape[1] - (self.muks.shape[1] - self.index - 1)
        if n_cols > 0:
            pad = np.full(shape=(self.B, np.abs(n_cols)), fill_value=np.nan)
            self.muks = np.hstack((self.muks, pad))
        elif n_cols < 0:
            pad = np.full(shape=(inds.size, np.abs(n_cols)), fill_value=np.nan)
            new_muks = np.hstack((new_muks, pad))

        ## incorporate new data
        # remove all data up to current index
        self.muks = self.muks[:, self.index :]
        # merge new data at starting at next index
        self.muks[inds, 1:] = new_muks
        self.tfs[inds] = new_tfs
        self.ruls[inds] = self.tfs[inds]  # - res_vec
        self.index = 0
        return

    def Pr_smaller_thresh(
        self,
        muks: np.ndarray,
        ub: float,
        p_thres: float,
    ) -> np.ndarray:
        """
        Checks whether the probability of failure until the next time step is bigger
        than a prescribed threshold.

        Args:
            muks: location parameter of the predicted lognormal distribution
            ub: upper bound for the CDF
            p_thres: failure probability threshold

        Returns:
            a: 0/1 array, where 1 represents a bigger failure probability than the threshold.
        """
        a = np.zeros_like(muks, dtype=int)  # default is do nothing
        # compute probability of failure within next timestep
        pf_pred = self.RUL_pred_k_cdf(x=ub, muks=muks)
        a[pf_pred > p_thres] = 1
        return a
