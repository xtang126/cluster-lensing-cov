#!/usr/bin/env python
"""
Correlated-LSS ("correlated haloes") covariance of a stacked DeltaSigma (Gruen+15 C^corr), for use with clens.

Gruen, Seitz, Becker, Friedrich & Mana 2015, MNRAS 449, 4264 (arXiv:1501.01632), Sec. 3.2.2, Eqs. 22-26.
The number of haloes h = (M, R_off) clustered around the cluster is Poisson, Cov(lambda_k, lambda_l) =
delta_kl <lambda_k>, so
      C^corr_ij = int dP_c(h) DS_i(h) DS_j(h)                                              (Eq. 24)
      dP_c = b(M_cl) b(M) n(M) dlnM  w_p(R_off) 2 pi R_off dR_off                           (Eqs. 25-26)
      w_p(R) = int_{-Pi}^{Pi} xi_lin(sqrt(R^2 + chi^2)) dchi,   Pi = 200 h^-1 Mpc
with only the *excess* (correlated) haloes: the uncorrelated "1" of 1+xi is the uncorrelated LSS that clens
already has. DS_i(h) is the annulus-averaged tangential DeltaSigma that one NFW halo (truncated at r200m,
Duffy+08 c(M)) at projected offset R_off produces around the cluster centre:
      DS_t(R) = M_cyl(<R)/(pi R^2) - Sigma_ring(R)   (exact for any mass distribution).
No mean is subtracted (Poisson term). b(M_cl) is the mean bias of the bin, since C^corr is linear in it.

Gruen+15 call the analytic term a lower limit (no 3-point terms) and calibrate it on simulations with
c_corr = 5.0 +- 0.3 (z=0.25; 4.2 +- 1.1 at z=0.5). c_corr=1 gives the pure analytic model.
The neighbour halo is placed at the cluster redshift (same Sigma_crit), as in Gruen+15's +-200 h^-1 Mpc window.

What this does NOT double count with clens: clens's Gaussian (C^hh+1/n_h)(C^SigmaSigma+...) + (C^hSigma)^2
has uncorrelated LSS and the Gaussian sample variance of the *mean* 2-halo term, but not the Poisson
scatter of the number of clustered neighbours from cluster to cluster.

Internally h-free (comoving Mpc, Msun/pc^2) like clens; calc_cov returns a CovStackedDeltaSigma in
h-units, exactly like IntrinsicCov.

Usage:
    from clss_cov import CLSSCov
    cl = CLSSCov(co, sr, zh_min, zh_max, lam_min, lam_max, c_corr=5.0)   # c_corr=1 -> analytic lower limit
    # rp_edges must match cds.calc_cov(rp_min, rp_max, n_rp): exp(linspace(log rp_min, log rp_max, n_rp+1))
    res = cl.calc_cov(rp_edges, N, cov_stat=cds.cov_sum)   # CovStackedDeltaSigma (h-units)
    res.matrix("clss"), res.total()
    res_all = ic.calc_cov(rp_edges, N, cds.cov_sum).merge(he.calc_cov(rp_edges, N, cds.cov_sum), res)
"""
import numpy as np

from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
from clens.ying.halostat import HaloStat
from intrinsic_cov import c_duffy08, CovStackedDeltaSigma
from halo_ell_cov import bin_subgrid, binned_ds_from_mcyl
from nfw_profile import NfwProfile


def projected_corr(cf, ximm, R, Pi):
    """w_p(R) = 2 int_0^Pi xi_lin(sqrt(R^2+chi^2)) dchi [Mpc]; xi clamped to its innermost value below r_min."""
    chi = np.concatenate([[0.], np.geomspace(1e-4, Pi, 3000)])
    r = np.sqrt(R[:, None]**2 + chi[None, :]**2)
    lr = np.log(np.clip(r, cf.radius[0], cf.radius[-1]))
    xi = np.interp(lr, np.log(cf.radius), ximm)
    return 2. * np.trapz(xi, chi, axis=1)


def offset_nodes(rp_edges, d_max, n_per=6, d_min=1e-4):
    """Gauss-Legendre nodes/weights in ln d, one panel per annulus: a small halo's binned DS_t has a kink
    where d crosses an annulus edge, so the edges are panel breakpoints. int f dln d = sum(w f)."""
    bp = np.unique(np.concatenate([[d_min], rp_edges[rp_edges > d_min], [d_max]]))
    u, wu = np.polynomial.legendre.leggauss(n_per)
    lo, hi = np.log(bp[:-1])[:, None], np.log(bp[1:])[:, None]
    return np.exp((0.5 * (hi - lo) * (u + 1.) + lo).ravel()), (0.5 * (hi - lo) * wu).ravel()


def offset_profiles(nfw, d, Rsub, n_th=64):
    """Binned DS_t around the cluster from haloes at projected offsets d, vectorised over haloes.

    nfw  : NfwProfile, scalar or a vector of n_halo haloes (M_2D = analytic truncated NfwProfile.mass_2d)
    d    : (n_off,) projected offsets [Mpc];  Rsub : (n_bin, n_sub) annulus sub-grid
    returns (*halo_shape, n_off, n_bin) [Msun/pc^2]

    Cylinder mass inside radius R (cluster-centred) of a halo at offset d, by parts:
        M_cyl(R; d) = int_a^b M_2D(<s) (-df/ds) ds,  a = |R-d|, b = R+d,
    f(s) = arccos[(s^2+d^2-R^2)/(2sd)]/pi = fraction of the halo's ring s inside the circle.
    With s = a + (b-a)(1-cos th)/2 the endpoint singularities cancel:
        M_cyl = (1/pi) int_0^pi M_2D(s) (s^2 - d^2 + R^2) / (s sqrt((s+a)(s+b))) dth.
    """
    u, wu = np.polynomial.legendre.leggauss(n_th)
    th, wth = 0.5 * np.pi * (u + 1.), 0.5 * np.pi * wu
    R = Rsub.ravel()[None, :, None]
    dd = np.asarray(d, float)[:, None, None]
    a, b = np.abs(R - dd), R + dd
    s = np.maximum(a + 0.5 * (b - a) * (1. - np.cos(th)), 1e-12)          # (n_off, nR, n_th)
    kern = wth * (s**2 - dd**2 + R**2) / (s * np.sqrt((s + a) * (s + b))) / np.pi
    Mcyl = np.einsum("...ort,ort->...or", nfw.mass_2d(s), kern)             # (*halo, n_off, nR)
    return binned_ds_from_mcyl(Rsub, Mcyl.reshape(Mcyl.shape[:-1] + Rsub.shape))


class CLSSCov(object):
    """
    Correlated-LSS covariance (Gruen+15 C^corr) of the stacked DeltaSigma for one (z, lambda) bin.

    co, sr              : clens CosmoParameters and scaling relation (log-normal class, or HODRichness with hook enabled)
    zh_min, zh_max      : halo redshift bin (profiles evaluated at the midpoint)
    lam_min, lam_max    : richness bin (sets the mean cluster bias b_cl)
    c_corr              : amplitude rescaling (Gruen+15 fit 5.0 +- 0.3; 1 = analytic lower limit)
    Pi_h                : line-of-sight half-window [h^-1 Mpc] (Gruen+15: 200)
    M_min_h, M_max_h    : neighbour-halo mass range [h^-1 Msun] (Gruen+15: 1e8 - 10^15.5)
    cM                  : median c(M) of the neighbours: 'duffy08' or 'correa' (clens default)
    nM, n_per, n_sub, n_th : quadrature sizes (neighbour mass, offset nodes per annulus, annulus sub-grid,
                          angle); one broadcast of nM * n_per*(n_rp+2) * n_rp*n_sub * n_th points
                          (~1 GB peak for the defaults and 15 bins)

    The slow clens setup (xi_lin, mass function, bias, b_cl) is done once in __init__; calc_cov can then
    be called for different radial binnings / N.
    """
    def __init__(self, co, sr, zh_min, zh_max, lam_min, lam_max, c_corr=5.0, Pi_h=200.,
                 M_min_h=1e9, M_max_h=10**15.5, cM="duffy08", nM=40, n_per=8, n_sub=12, n_th=24):
        if cM not in ("duffy08", "correa"):
            raise ValueError(f"unknown cM relation {cM!r}")
        self.co, self.sr = co, sr
        self.zh_min, self.zh_max = zh_min, zh_max
        self.zh = 0.5 * (zh_min + zh_max)
        self.lam_min, self.lam_max = lam_min, lam_max
        self.c_corr, self.Pi = c_corr, Pi_h / co.h                     # Pi in h-free Mpc
        self.cM = cM
        self.n_per, self.n_sub, self.n_th = n_per, n_sub, n_th

        self.cf = CorrelationFunctions3D(z=self.zh, co=co, sr=sr, lambda_min=lam_min, lambda_max=lam_max)
        self.ximm = self.cf.xi_mm()

        # mean bias of the clusters in the bin (C^corr is linear in it)
        cf = self.cf
        w_cl = cf.dndlnM_arr * cf.rs.lnM_selection(cf.lnM_arr, self.zh)
        self.b_cl = np.trapz(w_cl * cf.bias_lnM_interp(cf.lnM_arr), cf.lnM_arr) / np.trapz(w_cl, cf.lnM_arr)

        # neighbour haloes: Tinker08 mass function and Tinker10 bias over a wide range (h-free masses)
        self.lnM = np.linspace(np.log(M_min_h / co.h), np.log(M_max_h / co.h), nM)
        M = np.exp(self.lnM)
        hs = HaloStat(cosmo=cf.cosmo_ying, z=self.zh, DELTA_HALO=200., rho_mean_0=cf.den.rho_mean_z(0.),
                      mass=M, dlogm=self.lnM[1] - self.lnM[0])
        self.dndlnM = hs.mass_function * M
        self.bias = hs.bias_function
        self.nfw = NfwProfile(M, self.c_median(M), rho_ref=cf.rho_mean_z)   # nM haloes, M200m

    def c_median(self, M):
        """median c200m at mass M [h-free Msun]"""
        if self.cM == "duffy08":
            return c_duffy08(M, self.zh, self.co.h, "200m")
        return self.cf.c200m_M200m_Correa(M)

    def calc_profiles(self, rp_edges):
        """
        binned DeltaSigma around the cluster from one neighbour halo, on the (M, R_off) grid:
            self.d, self.w_d : (n_off,) offsets [Mpc] and weights 2 pi d^2 w_p(d) dln d  (correlated
                               neighbours per unit dn/dlnM, before the b_cl b(M) factor)
            self.profiles    : (nM, n_off, n_rp)   [Msun/pc^2, h-free]
        """
        rp_edges = np.asarray(rp_edges, float)
        # one offset grid for all masses: beyond rp_max + r200 a halo is outside every annulus (P = 0)
        d, w_lnd = offset_nodes(rp_edges, rp_edges[-1] + self.nfw.r200.max(), self.n_per)
        self.d = d
        self.w_d = w_lnd * 2. * np.pi * d**2 * projected_corr(self.cf, self.ximm, d, self.Pi)
        self.profiles = offset_profiles(self.nfw, d, bin_subgrid(rp_edges, self.n_sub), self.n_th)
        return self.profiles

    def calc_cov(self, rp_edges, N, cov_stat):
        """
        rp_edges : radial bin edges [comoving Mpc, h-free], len n_rp+1 -- must match the clens binning,
                   i.e. exp(linspace(log(rp_min), log(rp_max), n_rp+1)) for cds.calc_cov(rp_min, rp_max, n_rp)
        N        : number of clusters in the stack (e.g. from ClusterCounts)
        cov_stat : clens Gaussian covariance, cds.cov_sum [(Msun/pc^2)^2, h-free]
        returns  : CovStackedDeltaSigma in h-units with cov_stat and cov_clss (also stored as self.result);
                   the contribution per neighbour mass (h-free, per cluster, without c_corr) is kept in
                   self.dcov_dlnM, so that cov_clss = c_corr * trapz(dcov_dlnM, lnM) / N

            cov_clss = c_corr * b_cl * int dlnM b(M) dn/dlnM int d^2R_off w_p(R_off) DS_i DS_j / N
        """
        P = self.calc_profiles(rp_edges)
        self.dcov_dlnM = (self.b_cl * (self.dndlnM * self.bias)[:, None, None]
                          * np.einsum("o,mor,mos->mrs", self.w_d, P, P))      # (nM, n_rp, n_rp)
        cov_clss = self.c_corr * np.trapz(self.dcov_dlnM, self.lnM, axis=0) / N

        rp_edges = np.asarray(rp_edges)
        self.result = CovStackedDeltaSigma.from_clens(
            self.co.h, np.sqrt(rp_edges[:-1] * rp_edges[1:]),
            z_bin=(self.zh_min, self.zh_max), lambda_bin=(self.lam_min, self.lam_max),
            cov_stat=cov_stat, cov_clss=cov_clss)
        return self.result
