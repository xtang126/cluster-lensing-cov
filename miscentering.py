#!/usr/bin/env python
"""
Miscentering for clens (hywu/cluster-lensing-cov) stacked profiles.

Model (DES Y1: McClintock+19 Sec. 5.2, arXiv:1805.00039; prior: Zhang+19, arXiv:1901.07119):
    Sigma_mis(R | R_mis) = (1/2pi) int_0^{2pi} Sigma( sqrt(R^2 + R_mis^2 + 2 R R_mis cos phi) ) dphi
    P(x | tau) = x/tau^2 exp(-x/tau),   x = R_mis / R_lambda,   R_lambda = (lambda/100)^0.2 h^-1 Mpc (physical)
    DeltaSigma_obs = (1 - f_mis) DeltaSigma_cen + f_mis DeltaSigma_mis
DES Y1 (Zhang+19): rho = 0.835 -> f_mis = 0.165, tau = 0.166.

Units follow clens: h-free, comoving Mpc, Msun/pc^2.

Usage:
    from miscentering import sigma_and_deltasigma, miscentered_DeltaSigma, R_lambda_comoving
    rp, Sig, DS = sigma_and_deltasigma(lp)                 # lp = clens LensingProfiles
    DS_obs, DS_mis, DS_cen = miscentered_DeltaSigma(rp, Sig, R_lambda_comoving(25, 0.275, co.h))
"""
import numpy as np
from scipy.interpolate import interp1d


def sigma_and_deltasigma(lp):
    """rp [cMpc], Sigma, DeltaSigma [Msun/pc^2] from a clens LensingProfiles object.
    (LensingProfiles.calc_Sigma is broken upstream: it still references self.nu.)"""
    from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
    from clens.ying.DeltaSigmaR import DeltaSigmaR
    import clens.util.constants as cn
    zh = 0.5 * (lp.zh_min + lp.zh_max)
    rho_mean = cn.rho_crit_with_h * lp.co.h**2 * lp.co.OmegaM
    cf3d = CorrelationFunctions3D(z=zh, co=lp.co, sr=lp.sr,
                                  lambda_min=lp.lambda_min, lambda_max=lp.lambda_max)
    rp, DS, xirp, _ = DeltaSigmaR(cf3d.radius, cf3d.xi_cm(), rp_max=150, rho=rho_mean)
    return rp, rho_mean * xirp * 1e-12, DS * 1e-12


def R_lambda_comoving(lam, z, h):
    """redMaPPer R_lambda = (lambda/100)^0.2 h^-1 Mpc (physical) -> h-free comoving Mpc."""
    return (lam / 100.)**0.2 / h * (1 + z)


def _delta_sigma(rp, Sig):
    """DeltaSigma = mean Sigma inside R - Sigma(R), trapezoid in R dR."""
    cum = np.concatenate([[0.], np.cumsum(0.5 * (Sig[1:] * rp[1:] + Sig[:-1] * rp[:-1]) * np.diff(rp))])
    return (Sig[0] * rp[0]**2 / 2 + cum) * 2 / rp**2 - Sig


def miscentered_DeltaSigma(rp, Sigma, R_lambda, tau=0.166, f_mis=0.165, n_mis=200, n_phi=128):
    """
    rp, Sigma : centred projected profile (excess over the mean is fine; it cancels in DeltaSigma)
    R_lambda  : redMaPPer radius, same units as rp (use R_lambda_comoving for clens)
    returns   : DeltaSigma_obs, DeltaSigma_mis, DeltaSigma_cen
    """
    lnS = interp1d(np.log(rp), np.log(np.clip(Sigma, 1e-30, None)), bounds_error=False,
                   fill_value=(np.log(Sigma[0]), np.log(max(Sigma[-1], 1e-30))))
    S = lambda R: np.exp(lnS(np.log(np.clip(R, rp[0], rp[-1]))))

    x = np.linspace(1e-4, 12 * tau, n_mis)                 # x = R_mis / R_lambda
    px = x / tau**2 * np.exp(-x / tau)
    px /= np.trapz(px, x)
    Rmis = (x * R_lambda)[None, :, None]
    phi = np.linspace(0, np.pi, n_phi)[None, None, :]       # symmetric in phi -> [0, pi]

    R = rp[:, None, None]
    ring = S(np.sqrt(R**2 + Rmis**2 + 2 * R * Rmis * np.cos(phi)))
    Sig_mis = np.trapz(np.trapz(ring, phi[0, 0], axis=2) / np.pi * px[None, :], x, axis=1)

    DS_mis = _delta_sigma(rp, Sig_mis)
    DS_cen = _delta_sigma(rp, Sigma)
    return (1 - f_mis) * DS_cen + f_mis * DS_mis, DS_mis, DS_cen
