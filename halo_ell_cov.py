#!/usr/bin/env python
"""
Halo ellipticity + orientation covariance of a stacked DeltaSigma (Gruen+15 C^ell), for use with clens.

Gruen, Seitz, Becker, Friedrich & Mana 2015, MNRAS 449, 4264 (arXiv:1501.01632), Sec. 3.2.4, Eqs. 27-31:
  prolate NFW, volume-preserving ellipsoidal radius (long axis = z')
      r_e^2 = x^T A x,   A = diag(q^{-2/3}, q^{-2/3}, q^{4/3})                       (Eq. 27)
  axis ratio  P(q) ~ N(0.6, 0.12) truncated to 0.1 <= q <= 1  (Bett+07, ~1e14 Msun)    (Eq. 31)
  orientation isotropic: cos(alpha) ~ U[0, 1]
      C^ell_ij = E_{q,alpha}[DS_i DS_j] - E[DS_i] E[DS_j]                            (Eq. 30)
  Gruen+15 calibrate the analytic term on simulations with an amplitude c_ell = 3.7 +- 0.3 (z=0.25);
  c_ell=1 gives the pure analytic model.

Projection (exact for any ellipsoidal rho(r_e)): with line of sight n = (sin a, 0, cos a),
      Sigma(R) = Sigma_sph(xi) / sqrt(n.A.n),   xi^2 = R^T B R,
      B = diag(q^{2/3} / n.A.n, q^{-2/3})   (in the sky basis e1 = (cos a, 0, -sin a), e2 = y),
where Sigma_sph is the projection of the *spherical* profile rho(r). Projected mass is conserved.
The NFW is truncated at r_e = r200m; Sigma_sph is analytic (nfw_profile.NfwProfile.sigma_truncated,
Takada & Jain 2003 Eq. 27). Only the 1-halo term is elliptical; the (spherical, identical)
2-halo term cancels in the covariance. The tangential shear is azimuthally averaged, so
DS_t(R) = M_cyl(<R)/(pi R^2) - Sigma_ring(R), averaged over each annulus with area weighting; the
annulus average needs only M_cyl, which is closed-form up to one smooth angular average:
      M_cyl(<R) = (1/sqrt(n.A.n)) < M_2D(<R s(phi)) / s(phi)^2 >_phi,   s^2 = B11 cos^2 phi + B22 sin^2 phi,
(substitute xi = r s(phi) in int_0^R r dr Sigma_sph(r s)), with M_2D the analytic truncated-NFW
cylinder mass (NfwProfile.mass_2d). Everything is evaluated in one broadcast over
(mass, q, alpha, R, phi) -- no Python loops.

Within-mass variance only: the covariance is averaged over the bin's mass distribution
(dn/dlnM x P(lambda in bin | M)); the scatter *between* masses is in intrinsic_cov.py.

Internally h-free (comoving Mpc, Msun/pc^2) like clens; calc_cov returns a CovStackedDeltaSigma in
h-units, exactly like IntrinsicCov.

Usage:
    from halo_ell_cov import HaloEllCov
    he = HaloEllCov(co, sr, zh_min, zh_max, lam_min, lam_max, c_ell=3.7)   # c_ell=1 -> analytic only
    # rp_edges must match cds.calc_cov(rp_min, rp_max, n_rp): exp(linspace(log rp_min, log rp_max, n_rp+1))
    res = he.calc_cov(rp_edges, N, cov_stat=cds.cov_sum)   # CovStackedDeltaSigma (h-units)
    res.matrix("ell"), res.total()
    res_all = ic.calc_cov(rp_edges, N, cds.cov_sum).merge(res)   # with IntrinsicCov's components
"""
import numpy as np

from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
from intrinsic_cov import c_duffy08, CovStackedDeltaSigma
from nfw_profile import NfwProfile


def bin_subgrid(edges, n_sub=32):
    """Log-spaced sub-grid inside each annulus, shape (n_bin, n_sub)."""
    return np.exp(np.linspace(np.log(edges[:-1]), np.log(edges[1:]), n_sub, axis=1))


def binned_ds_from_mcyl(Rsub, Mcyl):
    """Annulus-averaged tangential DeltaSigma from the cylinder mass M_cyl(<R) [Msun/pc^2].
    <DS>_bin = [2 int M_cyl dlnR - (M_cyl(R2) - M_cyl(R1))] / (pi (R2^2 - R1^2)).
    Rsub (n_bin, n_sub) [Mpc]; Mcyl (..., n_bin, n_sub) [Msun]."""
    lnR = np.log(Rsub)
    I = np.trapz(Mcyl, lnR, axis=-1)
    dM = Mcyl[..., -1] - Mcyl[..., 0]
    area = np.pi * (Rsub[:, -1]**2 - Rsub[:, 0]**2)
    return (2. * I - dM) / area * 1e-12


def _q_alpha_nodes(q_mean, q_sigma, q_min, n_q, n_a):
    uq, wq = np.polynomial.legendre.leggauss(n_q)
    q = q_min + 0.5 * (uq + 1.) * (1. - q_min)
    wq = wq * np.exp(-0.5 * ((q - q_mean) / q_sigma)**2)
    ua, wa = np.polynomial.legendre.leggauss(n_a)
    cosa = 0.5 * (ua + 1.)
    Q, CA = np.meshgrid(q, cosa, indexing="ij")
    W = np.outer(wq, wa)
    return Q.ravel(), CA.ravel(), (W / W.sum()).ravel()


def ell_profiles(nfw, rp_edges, q, cosa, n_sub=24, n_phi=16):
    """Binned DS_t of prolate truncated NFWs, vectorised over haloes and orientations.

    nfw      : NfwProfile, scalar or a vector of n_halo haloes
    q, cosa  : (n_draw,) axis ratios and cos(alpha) of the line of sight to the long axis
    returns  : (*halo_shape, n_draw, n_bin)   [Msun/pc^2]
    The phi average uses the midpoint rule on a quarter period (s^2 has period pi and is even about
    0 and pi/2), which converges exponentially for this smooth periodic integrand.
    """
    Rsub = bin_subgrid(np.asarray(rp_edges, float), n_sub)                    # (n_bin, n_sub)
    phi = (np.arange(n_phi) + 0.5) / n_phi * 0.5 * np.pi
    nAn = q**(-2. / 3.) * (1. - cosa**2) + q**(4. / 3.) * cosa**2             # (n_draw,)
    s2 = (q**(2. / 3.) / nAn)[:, None] * np.cos(phi)**2 + q[:, None]**(-2. / 3.) * np.sin(phi)**2
    x = Rsub[None, :, :, None] * np.sqrt(s2)[:, None, None, :]                # (n_draw, n_bin, n_sub, n_phi)
    M2 = nfw.mass_2d(x)                                                       # (*halo, n_draw, n_bin, n_sub, n_phi)
    Mcyl = (M2 / s2[:, None, None, :]).mean(axis=-1) / np.sqrt(nAn)[:, None, None]
    return binned_ds_from_mcyl(Rsub, Mcyl)


class HaloEllCov(object):
    """
    Halo ellipticity + orientation covariance (Gruen+15 C^ell) of the stacked DeltaSigma for one (z, lambda) bin.

    co, sr              : clens CosmoParameters and scaling relation (log-normal class, or HODRichness with hook enabled)
    zh_min, zh_max      : halo redshift bin (profiles evaluated at the midpoint)
    lam_min, lam_max    : richness bin
    c_ell               : amplitude rescaling (Gruen+15 fit 3.7 +- 0.3; 1 = analytic model)
    q_mean, q_sigma, q_min : prolate axis-ratio distribution, N(q_mean, q_sigma) on [q_min, 1] (Gruen+15 Eq. 31)
    cM                  : median c(M): 'duffy08' (Duffy+08 c200m(M,z)) or 'correa' (clens default)
    n_q, n_a            : Gauss-Legendre nodes in q and cos(alpha)
    nM, wcut            : mass nodes, mass-weight cut (relative to peak) -- same mass grid as IntrinsicCov
    n_sub, n_phi        : annulus sub-grid points, quarter-circle angle points

    The slow clens setup (mass function, selection) is done once in __init__; calc_cov can then be
    called for different radial binnings / N.
    """
    def __init__(self, co, sr, zh_min, zh_max, lam_min, lam_max, c_ell=3.7,
                 q_mean=0.6, q_sigma=0.12, q_min=0.1, cM="duffy08",
                 n_q=12, n_a=12, nM=12, wcut=1e-3, n_sub=24, n_phi=16):
        if cM not in ("duffy08", "correa"):
            raise ValueError(f"unknown cM relation {cM!r}")
        self.co, self.sr = co, sr
        self.zh_min, self.zh_max = zh_min, zh_max
        self.zh = 0.5 * (zh_min + zh_max)
        self.lam_min, self.lam_max = lam_min, lam_max
        self.c_ell, self.cM = c_ell, cM
        self.n_sub, self.n_phi = n_sub, n_phi

        self.cf = CorrelationFunctions3D(z=self.zh, co=co, sr=sr, lambda_min=lam_min, lambda_max=lam_max)

        # mass weights of clusters in the bin, on clens's fine grid -> coarse grid covering the support
        cf = self.cf
        w_fine = cf.dndlnM_arr * cf.rs.lnM_selection(cf.lnM_arr, self.zh)
        keep = w_fine > wcut * w_fine.max()
        self.lnM = np.linspace(cf.lnM_arr[keep][0], cf.lnM_arr[keep][-1], nM)
        w = np.interp(self.lnM, cf.lnM_arr, w_fine)
        self.w_M = w / w.sum()

        # one NfwProfile holding all nM haloes (M200m, same definition as clens NFW)
        M = np.exp(self.lnM)
        self.nfw = NfwProfile(M, self.c_median(M), rho_ref=cf.rho_mean_z)

        # shape/orientation nodes: q on [q_min, 1] x cos(alpha) on [0, 1]
        self.q, self.cosa, self.w_qa = _q_alpha_nodes(q_mean, q_sigma, q_min, n_q, n_a)

    def c_median(self, M):
        """median c200m at mass M [h-free Msun]"""
        if self.cM == "duffy08":
            return c_duffy08(M, self.zh, self.co.h, "200m")            # clens halos are M200m
        return self.cf.c200m_M200m_Correa(M)

    def calc_profiles(self, rp_edges):
        """
        binned DeltaSigma of the prolate 1-halo term for every draw, on the (M, (q, alpha)) grid:
            self.profiles : (nM, n_q * n_a, n_rp)   [Msun/pc^2, h-free]
        """
        self.profiles = ell_profiles(self.nfw, np.asarray(rp_edges, float), self.q, self.cosa,
                                     n_sub=self.n_sub, n_phi=self.n_phi)
        return self.profiles

    def calc_cov(self, rp_edges, N, cov_stat):
        """
        rp_edges : radial bin edges [comoving Mpc, h-free], len n_rp+1 -- must match the clens binning,
                   i.e. exp(linspace(log(rp_min), log(rp_max), n_rp+1)) for cds.calc_cov(rp_min, rp_max, n_rp)
        N        : number of clusters in the stack (e.g. from ClusterCounts)
        cov_stat : clens Gaussian covariance, cds.cov_sum [(Msun/pc^2)^2, h-free]
        returns  : CovStackedDeltaSigma in h-units with cov_stat and cov_ell (also stored as self.result);
                   the per-mass covariances (h-free, per cluster, without c_ell) are kept in self.cov_M

            cov_ell = c_ell * E_M[ Cov_{q,alpha}( DS | M ) ] / N
        """
        P = self.calc_profiles(rp_edges)
        wqa = self.w_qa
        mean_M = np.einsum("d,mdr->mr", wqa, P)                           # E(DS | M)
        dP = P - mean_M[:, None, :]
        self.cov_M = np.einsum("d,mdr,mds->mrs", wqa, dP, dP)              # Cov_{q,alpha}(DS | M)
        cov_ell = self.c_ell * np.einsum("m,mrs->rs", self.w_M, self.cov_M) / N

        rp_edges = np.asarray(rp_edges)
        self.result = CovStackedDeltaSigma.from_clens(
            self.co.h, np.sqrt(rp_edges[:-1] * rp_edges[1:]),
            z_bin=(self.zh_min, self.zh_max), lambda_bin=(self.lam_min, self.lam_max),
            cov_stat=cov_stat, cov_ell=cov_ell)
        return self.result
