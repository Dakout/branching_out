import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.stats as sps
from eraUQ import ERADist


class Simulator:
    def __init__(self, cf: dict) -> None:
        self.cf = cf
        np.random.seed(self.cf["rng_seed"])
        self.sigexp = np.exp(0.5 * self.cf["sigma_lnerr"] ** 2)
        self.dist_object = ERADist(
            cf["dist_type"], "MOM", [cf["mu_TF"], cf["sigma_TF"]]
        )
        return

    def draw_tf_samples(self, Ns: int) -> np.ndarray:
        return self.dist_object.random(size=Ns)

    def draw_tf_samples_bigger_val(self, Ns: int, val: float) -> np.ndarray:
        tfs = self.draw_tf_samples(Ns=Ns)
        inds = tfs < val
        while inds.any():
            tfs[inds] = self.draw_tf_samples(Ns=inds.sum())
            inds = tfs < val
        return tfs

    def draw_tf_samples_bigger_val_sym(self, Ns: int, val) -> np.ndarray:
        tfs = self.draw_tf_samples(Ns=Ns)
        inds = np.logical_or(tfs < val, tfs > 2 * self.cf["mu_TF"] - val)
        while inds.any():
            tfs[inds] = self.draw_tf_samples(Ns=inds.sum())
            inds = np.logical_or(tfs < val, tfs > 2 * self.cf["mu_TF"] - val)
        return tfs

    def decision_times_given_tf(
        self, tf_sample: np.ndarray, res_time: float | None = None
    ) -> np.ndarray:
        if res_time is None:
            res_time = self.cf["Delta_T"]
        return np.arange(res_time, tf_sample, self.cf["Delta_T"])

    def corr_matrix_given_tk(self, tk: np.ndarray) -> np.ndarray:
        R = np.zeros((len(tk), len(tk)))
        # correlation model
        for i in range(len(tk)):
            R[i, :] = np.exp(-abs(tk[i] - tk) / self.cf["corr_len"])
        return R

    def cov_matrix_given_R(self, R: np.ndarray) -> np.ndarray:
        D = np.eye(R.shape[0]) * self.cf["sigma_lnerr"]
        C = D @ R @ D
        return C

    def draw_lnerr_sample_given_C(self, C: np.ndarray, size: int = 1) -> np.ndarray:
        ln_e_k = sps.multivariate_normal.rvs(
            mean=np.zeros(C.shape[0]), cov=C, size=size
        )
        return ln_e_k

    def muks_given_tf(
        self, tf_sample: np.ndarray, res_time: float | None = None
    ) -> np.ndarray:
        tk = self.decision_times_given_tf(tf_sample=tf_sample, res_time=res_time)
        R = self.corr_matrix_given_tk(tk=tk)
        C = self.cov_matrix_given_R(R=R)
        ln_e_k = self.draw_lnerr_sample_given_C(C=C)
        muks = np.log(tf_sample - tk) + ln_e_k
        return muks

    def RUL_pred_k_pdf(self, x: np.ndarray, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.pdf(x, s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_cdf(self, x: np.ndarray, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.cdf(x, s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_icdf(self, q: np.ndarray, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.ppf(q=q, s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_ci(self, muks: np.ndarray, alpha: float = 0.99) -> np.ndarray:
        ci_lower = np.array(
            [
                sps.lognorm.ppf(
                    (1 - alpha) / 2,
                    s=self.cf["sigma_lnerr"],
                    scale=np.exp(muks[m]),
                )
                for m in muks
            ]
        )
        ci_upper = np.array(
            [
                sps.lognorm.ppf(
                    1 - (1 - alpha) / 2,
                    s=self.cf["sigma_lnerr"],
                    scale=np.exp(muks[m]),
                )
                for m in muks
            ]
        )
        return ci_lower, ci_upper

    def RUL_pred_k_mean(self, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.mean(s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_median(self, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.median(s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_var(self, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.var(s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_std(self, muks: np.ndarray) -> np.ndarray:
        return sps.lognorm.std(s=self.cf["sigma_lnerr"], scale=np.exp(muks))

    def RUL_pred_k_ci2(self, muks: np.ndarray, alpha: float = 0.99) -> np.array:
        return sps.lognorm.interval(
            confidence=alpha, s=self.cf["sigma_lnerr"], scale=np.exp(muks)
        )

    def RUL_pred_k_rvs(self, muks: np.ndarray, size: int = 1) -> np.ndarray:
        return sps.lognorm.rvs(s=self.cf["sigma_lnerr"], scale=np.exp(muks), size=size)

    def cond_RUL_smaller(self, muks: np.ndarray, iub: float) -> np.ndarray:
        frac = (np.log(iub) - muks) / self.cf["sigma_lnerr"]
        return (
            self.sigexp * np.exp(muks) * sps.norm.cdf(x=frac - self.cf["sigma_lnerr"])
        )

    def cond_RUL_bigger(self, muks: np.ndarray, ilb: float) -> np.ndarray:
        frac = (muks - np.log(ilb)) / self.cf["sigma_lnerr"]
        return (
            self.sigexp * np.exp(muks) * sps.norm.cdf(x=frac + self.cf["sigma_lnerr"])
        )

    def cond_RUL_betw(self, muks: np.ndarray, ilb: float, iub: float) -> np.ndarray:
        frac_lb = (np.log(ilb) - muks) / self.cf["sigma_lnerr"]
        frac_ub = (np.log(iub) - muks) / self.cf["sigma_lnerr"]
        return (
            self.sigexp
            * np.exp(muks)
            * (
                sps.norm.cdf(x=frac_ub - self.cf["sigma_lnerr"])
                - sps.norm.cdf(x=frac_lb - self.cf["sigma_lnerr"])
            )
        )

    def get_max_len(self, muks: list) -> int:
        max_len = 0
        for m in muks:
            if max_len < m.size:
                max_len = m.size
        return max_len

    def nan_pad(self, muks: list, max_len: int) -> list:
        for k in range(len(muks)):
            if muks[k].size != max_len:
                muks[k] = np.concatenate(
                    (muks[k], np.full((max_len - muks[k].size,), np.nan))
                )
        return muks

    def muks_given_tf_vec(
        self,
        tf_vec: np.ndarray,
        get_npvec: bool = True,
        res_vec: np.ndarray | None = None,
    ) -> list | np.ndarray:
        mu_list = []
        if res_vec is None:
            # for k in tqdm(range(len(tf_vec)), mininterval=50):
            for k in range(tf_vec.size):
                mu_list.append(self.muks_given_tf(tf_sample=tf_vec[k]))
        else:
            # for k in tqdm(range(len(tf_vec)), mininterval=50):
            for k in range(tf_vec.size):
                mu_list.append(
                    self.muks_given_tf(tf_sample=tf_vec[k], res_time=res_vec[k])
                )
        # convert to numpy
        if get_npvec:
            max_len = self.get_max_len(muks=mu_list)
            mu_list = self.nan_pad(muks=mu_list, max_len=max_len)
            mu_list = np.vstack(mu_list)
        return mu_list

    ##################################################################################
    ################ Functions for cinf initialization ###############################
    ##################################################################################

    ############ Maintenance functions based on underlying TF-distribution ###########

    def cinf_wrt_tf_given_trep(self, t_rep: float) -> float:
        """
        Analytical target function for init option 1 (age-based replacement).
        """
        Pf = self.dist_object.cdf(x=t_rep)
        mu_cm = self.cf["c_c"] * Pf + (self.cf["c_f"] + self.cf["c_v"]) * (1 - Pf)
        mu_tlc = spi.quad(
            lambda t: t * self.dist_object.pdf(x=t),
            a=self.cf["mu_TF"] - self.cf["sig_int_factor"] * self.cf["sigma_TF"],
            b=t_rep,
        )[
            0
        ] + t_rep * (  # * Pf
            1 - Pf
        )
        return mu_cm / mu_tlc

    def cinf_init_opt1(self, return_res: bool = False) -> float:
        """
        Estimation of cinf based on an age-based replacement policy.
        """
        res = spo.minimize_scalar(
            fun=self.cinf_wrt_tf_given_trep,
            method="bounded",
            bounds=(
                0,
                self.cf["mu_TF"] + 3 * self.cf["sigma_TF"],
            ),
        )
        if return_res:
            return res
        else:
            return res.fun

    def cinf_init_opt2(self) -> float:
        """
        Estimation of cinf based on a perfect replacement policy.
        """
        return (self.cf["c_f"] + self.cf["c_v"]) / self.cf["mu_TF"]

    def cinf_init_opt3(self, fac: float = 0.5) -> float:
        """
        Estimation of cinf based on an interpolation of options 1 and 2.
        """
        return fac * self.cinf_init_opt1() + (1 - fac) * self.cinf_init_opt2()

    def cinf(self, fun: str, fac: float = 0.5) -> float:
        """
        Main initialization function, calls the right functions
        based on the passed options.
        """
        f_dict = {
            "opt1": self.cinf_init_opt1(),
            "opt2": self.cinf_init_opt2(),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.cinf_init_opt3(fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.cinf_init_opt1()

    ######### Maintenance functions based on a subset of tf-samples ######################

    def cinf_wrt_tf_given_trep_sample_based(
        self, t_rep: float, tfs: np.ndarray
    ) -> float:
        """
        Sample-based target function for init option 1 (age-based replacement).
        """
        lc_vec = np.zeros_like(tfs)
        cost_vec = np.zeros_like(tfs)
        fail_ind = tfs < t_rep

        lc_vec[fail_ind] = tfs[fail_ind]
        lc_vec[~fail_ind] = t_rep

        cost_vec[fail_ind] = self.cf["c_c"]
        cost_vec[~fail_ind] = self.cf["c_f"] + self.cf["c_v"]

        # print(t_rep, fail_ind, lc_vec, cost_vec)
        return np.mean(cost_vec) / np.mean(lc_vec)

    def cinf_init_opt1_sample_based(self, tfs: np.ndarray) -> float:
        """
        Estimation of cinf based on an age-based replacement policy.
        """
        t_check = np.linspace(1, self.cf["mu_TF"] + 5 * self.cf["sigma_TF"], int(5e4))
        res = np.array(
            [
                self.cinf_wrt_tf_given_trep_sample_based(t_rep=t, tfs=tfs)
                for t in t_check
            ]
        )
        return np.min(res)

    def cinf_init_opt2_sample_based(self, tfs: np.ndarray) -> float:
        """
        Estimation of cinf based on a perfect replacement policy.
        """
        return (self.cf["c_f"] + self.cf["c_v"]) / np.mean(tfs)

    def cinf_init_opt3_sampled_based(self, tfs: np.ndarray, fac: float = 0.5) -> float:
        """
        Estimation of cinf based on an interpolation of options 1 and 2.
        """
        res = fac * self.cinf_init_opt1_sample_based(tfs=tfs) + (
            1 - fac
        ) * self.cinf_init_opt2_sample_based(tfs=tfs)
        return res

    def cinf_sample_based(
        self, fun: str, tfs: np.ndarray, fac: float | None = None
    ) -> float:
        """
        Main initialization function, calls the right functions
        based on the passed options.
        """
        f_dict = {
            "opt1": self.cinf_init_opt1_sample_based(tfs=tfs),
            "opt2": self.cinf_init_opt2_sample_based(tfs=tfs),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.cinf_init_opt3_sampled_based(tfs=tfs, fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.cinf_init_opt1_sample_based(tfs=tfs)
