#!/usr/bin/env python
"""
HOD-style richness-mass relation for clens (hywu/cluster-lensing-cov).

Replaces clens's log-normal P(ln lambda | M) with the occupation model of
Costanzi et al. 2026 (arXiv:2604.05833), Sec. III, eq. 15:

    lambda_tr  = lambda_cen + lambda_sat
    lambda_cen = 1 if M >= Mmin else 0
    <lambda_sat|M,z> = ((M - Mmin)/(M1 - Mmin))^alpha * ((1+z)/(1+z*))^eps
    P(lambda_sat) = Poisson(<lambda_sat>)  (x)  Gaussian(0, sigma_lambda * <lambda_sat>)

Defaults = C26 mock values (Sec. III A): Mmin=10^11.4, M1=10^12.7 [Msun/h, M200m],
alpha=0.86, eps=0.28, sigma_lambda=0.185, z*=0.45.

On top of this true richness, P(lambda_ob | lambda_tr, z) (projection) is applied
following Costanzi et al. 2019 (arXiv:1807.07072), eq. 6 -- same model, same
calibration file (prj_params_DESY3_lss_lin_dep_getdist_v1.txt) and same
Gaussian-core + exponentially-modified-Gaussian (EMG) tail construction used in
mocks/mock_cluster_buzzard/src/costanzi_selection.py:

    P(lambda_ob|lambda_tr, z) = (1 - f_prj) * N(mu, sigma)
                              +  f_prj * EMG(mu, sigma, tau)

with (f_prj, tau, mu, sigma) functions of lambda_tr via (a, b) coefficients that
are themselves linearly interpolated in z across 15 knots on [0.10, 0.80]. The
intrinsic (Poisson + super-Poisson) scatter in lambda_tr is folded into the
Gaussian core's width before convolving with the projection kernel (see
_p_bin_obs below); set use_projection=False to recover the pre-projection TRUE
richness selection.

Usage:
    from hod_richness import HODRichness, enable_hod_hook
    enable_hod_hook()                 # once per session
    sr = HODRichness(h=co.h)          # pass wherever clens wants a scaling relation
"""
import os
import sys
import numpy as np
from scipy.stats import norm, exponnorm
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clens.util.scaling_relation as _srmod

# ---------------------------------------------------------------------------
# Costanzi+19 projection kernel: calibration file + per-coefficient z-interpolation
# (identical file and knot grid as mocks/mock_cluster_buzzard/src/costanzi_selection.py)
# ---------------------------------------------------------------------------
_PRJ_PARAM_FILE = os.path.join(os.path.dirname(__file__),
                                "prj_params_DESY3_lss_lin_dep_getdist_v1.txt")
_PRJ_Z_BINS = np.linspace(0.10, 0.80, 15)
_PRJ_LABELS = ["a_tau", "b_tau", "a_mu", "b_mu", "a_sig", "b_sig",
               "a_fprj", "b_fprj", "a_fmsk", "b_fmsk"]


def _load_prj_table(path=_PRJ_PARAM_FILE):
    params = np.loadtxt(path).T  # shape (10, 15)
    return {key: params[i, :] for i, key in enumerate(_PRJ_LABELS)}


def _prj_coeffs_at_z(z, table=None):
    """Linear-in-z interpolation of the 8 (a,b) coefficients actually used
    by the lambda_ob|lambda_tr kernel (a_fmsk/b_fmsk are a masking-fraction
    term, unused here, exactly as in the upstream projection_params())."""
    if table is None:
        table = _load_prj_table()
    needed = ["a_mu", "b_mu", "a_sig", "b_sig", "a_tau", "b_tau", "a_fprj", "b_fprj"]
    return {k: np.interp(z, _PRJ_Z_BINS, table[k]) for k in needed}


class HODRichness(object):
    def __init__(self, lgMmin=11.4, lgM1=12.7, alpha=0.86, eps=0.28, sigma_lambda=0.185,
                 zstar=0.45, h=0.7, nmax_pad=12, use_projection=True,
                 prj_param_file=_PRJ_PARAM_FILE):
        # C26 masses are in Msun/h; clens is h-free
        self.Mmin = 10**lgMmin / h
        self.M1 = 10**lgM1 / h
        self.alpha, self.eps, self.sig, self.zstar = alpha, eps, sigma_lambda, zstar
        self.nmax_pad = nmax_pad
        self.use_projection = use_projection
        self._prj_table = _load_prj_table(prj_param_file) if use_projection else None

    def mean_sat(self, lnM, z):
        M = np.exp(np.atleast_1d(lnM))
        x = np.clip((M - self.Mmin) / (self.M1 - self.Mmin), 0, None)
        return x**self.alpha * ((1 + z) / (1 + self.zstar))**self.eps

    def p_bin(self, lnM, z, lam_min, lam_max):
        """P(lam_min <= lambda_ob < lam_max | M, z): sum over Poisson satellite
        draws n of the true richness ltr=ncen+n, each convolved with the
        Costanzi+19 projection kernel P(lambda_ob|ltr,z) (or, if
        use_projection=False, just the intrinsic TRUE-richness bin probability)."""
        lnM = np.atleast_1d(lnM)
        mu = self.mean_sat(lnM, z)[:, None]
        ncen = (np.exp(lnM) >= self.Mmin).astype(float)[:, None]
        nmax = int(mu.max() + self.nmax_pad * np.sqrt(mu.max() + 1) + 20)
        n = np.arange(nmax + 1)[None, :]
        with np.errstate(divide="ignore"):
            pois = np.exp(n * np.log(np.where(mu > 0, mu, 1e-300)) - mu - gammaln(n + 1))
        w_intr = np.maximum(self.sig * mu, 1e-6)  # intrinsic (super-Poisson) scatter, C26

        if not self.use_projection:
            prob = pois * (norm.cdf((lam_max - ncen - n) / w_intr) -
                            norm.cdf((lam_min - ncen - n) / w_intr))
            return prob.sum(axis=1)

        p_given_n = self._p_bin_obs(ncen + n, w_intr, z, lam_min, lam_max)
        return (pois * p_given_n).sum(axis=1)

    def _p_bin_obs(self, ltr, w_intr, z, lam_min, lam_max):
        """P(lam_min <= lambda_ob < lam_max | lambda_tr, z), C19 eq.6, with the
        intrinsic Gaussian scatter w_intr around ltr folded into the kernel's
        Gaussian-core width (mu(ltr) is linear in ltr, so to leading order this
        is exact for the Gaussian branch and a good approximation for the EMG
        branch as long as w_intr << ltr, i.e. away from the M~Mmin threshold)."""
        c = _prj_coeffs_at_z(z, self._prj_table)
        ltr_eff = np.clip(ltr, 1.0, None)  # kernel is fit on real (>0) richness; guard ltr->0

        mu_pr = c["a_mu"] + c["b_mu"] * ltr_eff
        sigma_pr = c["b_sig"] * ltr_eff**c["a_sig"]
        tau_pr = np.clip(c["b_tau"] / ltr_eff**c["a_tau"], 1e-6, None)
        fprj = np.clip(c["b_fprj"] / (1. + np.exp(-ltr_eff))**c["a_fprj"], 0., 1.)
        sigma_eff = np.sqrt(sigma_pr**2 + (c["b_mu"] * w_intr)**2)

        cdf_gauss_hi = norm.cdf((lam_max - mu_pr) / sigma_eff)
        cdf_gauss_lo = norm.cdf((lam_min - mu_pr) / sigma_eff)

        K = np.clip(1. / (tau_pr * sigma_eff), 1e-6, None)  # scipy exponnorm shape param
        cdf_emg_hi = exponnorm.cdf(lam_max, K, loc=mu_pr, scale=sigma_eff)
        cdf_emg_lo = exponnorm.cdf(lam_min, K, loc=mu_pr, scale=sigma_eff)

        return (1. - fprj) * (cdf_gauss_hi - cdf_gauss_lo) + fprj * (cdf_emg_hi - cdf_emg_lo)

    def lnlambda_lnM(self, lnM, z):
        """Mean relation ln<lambda_tr> (TRUE richness, no projection), for
        plotting / comparison with log-normal relations."""
        lnM = np.atleast_1d(lnM)
        return np.log((np.exp(lnM) >= self.Mmin) + self.mean_sat(lnM, z) + 1e-30)


_orig_selection = _srmod.RichnessSelection.lnM_selection


def _hod_aware_selection(self, lnM, z):
    if hasattr(self.scaling_relation, "p_bin"):
        return self.scaling_relation.p_bin(lnM, z, self.lambda_min, self.lambda_max)
    return _orig_selection(self, lnM, z)


def enable_hod_hook():
    """Route clens's RichnessSelection through any scaling relation that has p_bin().
    Affects counts, bias (-> covariance) and the mean profile. Log-normal classes are unaffected."""
    _srmod.RichnessSelection.lnM_selection = _hod_aware_selection
