#!/usr/bin/env python
"""
Halo-to-halo ("intrinsic") covariance of a stacked DeltaSigma, for use with clens.

clens computes only the Gaussian-field covariance (Wu+19 eq. 22: shape noise + LSS + (C_ell^hSigma)^2).
It has NO term for the fact that the N stacked clusters have different profiles. That scatter comes from:
  (a) the spread of halo masses inside the richness bin  -> set by P(lambda|M)  (C21 log-normal or HOD)
  (b) random miscentering: which clusters are offset, and by how much  -> f_mis, tau
This module computes it the way the DES Y1 semi-analytic covariance did for these terms
(McClintock+19, arXiv:1805.00039, Sec. 3.2.4: mass-richness scatter and miscentering, by sampling the population):

    Cov_intr(R_i, R_j) = (1/N) * [ E_pop(DS_i DS_j) - E_pop(DS_i) E_pop(DS_j) ]

where E_pop averages over clusters in the bin, weighted by dn/dlnM * P(lambda in bin | M) and by the
miscentering mixture (1-f_mis) delta(R_mis) + f_mis * Gamma(x|tau).
  (c) concentration scatter at fixed mass  -> log-normal around a c(M) relation, sigma_lnc
Each cluster's profile follows clens: NFW + b(M) xi_lin, max(1h, 2h), but the NFW concentration is drawn from
the Duffy et al. 2008 (arXiv:0804.2486) c(M) relation by default (cM='duffy08'; cM='correa' reproduces clens):

    c200m(M, z) = A (M / M_pivot)^B (1+z)^C,   M_pivot = 2e12 h^-1 Msun
    full sample, Delta = 200 x mean, 0 < z < 2 (Duffy+08 Table 1):  A = 10.14, B = -0.081, C = -1.01
    scatter at fixed M: sigma_log10(c) ~ 0.15 (full sample)  ->  sigma_lnc = 0.15 ln10 ~ 0.35

The scatter is integrated with Gauss-Hermite quadrature in ln c (n_c nodes);
include_concentration=False turns it off (median c(M) only).
Not here (separate calculators, same CovStackedDeltaSigma result, combine with .merge):
  halo ellipticity/orientation -> halo_ell_cov.HaloEllCov (cov_ell, Gruen+15 C^ell);
  correlated LSS (Poisson scatter of clustered neighbour haloes, not in the clens Gaussian part)
                               -> clss_cov.CLSSCov (cov_clss, Gruen+15 C^corr).

Usage:
    from intrinsic_cov import IntrinsicCov
    ic = IntrinsicCov(co, sr, zh_min, zh_max, lam_min, lam_max,
                      f_mis=0.165, tau=0.166,                # f_mis=0 -> no miscentering
                      include_concentration=True,            # False -> no c scatter
                      cM='duffy08', sigma_lnc=0.35)
    # rp_edges must match cds.calc_cov(rp_min, rp_max, n_rp): exp(linspace(log rp_min, log rp_max, n_rp+1))
    res = ic.calc_cov(rp_edges, N, cov_stat=cds.cov_sum)   # CovStackedDeltaSigma, in h-units (Mpc/h, h Msun/pc^2)
    res.matrix("cM"), res.matrix("lambdaM"), res.matrix("miscen"), res.matrix("stat")
    res.total(), res.total(include=["lambdaM", "cM", "miscen"])   # full / intrinsic only
    for k, C in res.components().items(): plt.plot(res.R_hiMpc, np.sqrt(np.diag(C)), label=res.label(k))
    res.save("cov.npz"); CovStackedDeltaSigma.load("cov.npz")
    # all terms in one object (cov_stat is passed to each and must agree):
    res = res.merge(HaloEllCov(co, sr, ...).calc_cov(rp_edges, N, cds.cov_sum),
                    CLSSCov(co, sr, ...).calc_cov(rp_edges, N, cds.cov_sum))
"""
import numpy as np
from scipy.interpolate import interp1d

import clens.util.constants as cn
from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
from clens.ying.DeltaSigmaR import DeltaSigmaR
from miscentering import R_lambda_comoving
from clens.ying.nfw import NFW
from dataclasses import dataclass, field, fields

# Duffy+08 Table 1, NFW, full sample, 0<z<2
DUFFY08 = {"200m": (10.14, -0.081, -1.01),
           "vir":  (7.85,  -0.081, -0.71),
           "200c": (5.71,  -0.084, -0.47)}


def c_duffy08(M, z, h, mdef="200m"):
    """Duffy+08 median concentration; M in h-free Msun (converted to the h^-1 Msun pivot)."""
    A, B, C = DUFFY08[mdef]
    return A * (M * h / 2e12)**B * (1. + z)**C


def _ring_average(rp, Sig, R, Rmis, n_phi=96):
    """Sigma at radius R measured from a centre offset by each Rmis -> shape (len(Rmis), len(R))."""
    lnS = interp1d(np.log(rp), np.log(np.clip(Sig, 1e-30, None)), bounds_error=False,
                   fill_value=(np.log(max(Sig[0], 1e-30)), np.log(max(Sig[-1], 1e-30))))
    phi = np.linspace(0, np.pi, n_phi)
    d = np.sqrt(R[None, :, None]**2 + Rmis[:, None, None]**2
                + 2 * R[None, :, None] * Rmis[:, None, None] * np.cos(phi)[None, None, :])
    S = np.exp(lnS(np.log(np.clip(d, rp[0], rp[-1]))))
    return np.trapz(S, phi, axis=2) / np.pi


def _delta_sigma(R, Sig):
    cum = np.concatenate([[0.], np.cumsum(0.5 * (Sig[..., 1:] * R[1:] + Sig[..., :-1] * R[:-1]) * np.diff(R), axis=-1)],
                         axis=-1) if Sig.ndim == 1 else \
        np.concatenate([np.zeros(Sig.shape[:-1] + (1,)),
                        np.cumsum(0.5 * (Sig[..., 1:] * R[1:] + Sig[..., :-1] * R[:-1]) * np.diff(R), axis=-1)], axis=-1)
    return (Sig[..., :1] * R[0]**2 / 2 + cum) * 2 / R**2 - Sig


def _bin_average(R, DS, edges):
    """Area-weighted average of DS over annuli [edges[i], edges[i+1]]."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (R >= lo) & (R <= hi)
        out.append(np.trapz(DS[..., m] * R[m], R[m], axis=-1) / np.trapz(R[m], R[m]))
    return np.stack(out, axis=-1)


@dataclass(frozen=True, eq=False)
class CovStackedDeltaSigma:
    """Covariance of DeltaSigma(R) in one (z, lambda) bin.

    NOTE: R in comoving Mpc/h; DeltaSigma in h M_sun/pc^2,
    so every component is in (h M_sun/pc^2)^2.
    (clens works h-free; use from_clens / IntrinsicCov.calc_cov, which convert.)
    Components left as None were not computed (e.g. no c scatter, f_mis=0).
    """
    R_hiMpc: np.ndarray
    z_bin: tuple[float, float]
    lambda_bin: tuple[float, float]

    # ---- components ----
    cov_stat: np.ndarray = field(
        metadata={"component": True, "label": "shape noise + LSS"})        # Wu+19 eq. 22 (clens cov_sum)
    cov_cM: np.ndarray | None = field(
        default=None, metadata={"component": True, "label": r"$c$–$M$"})   # Duffy+08 c(M), log-normal scatter
    cov_lambdaM: np.ndarray | None = field(
        default=None, metadata={"component": True, "label": r"$\lambda$–$M$"})  # mass spread in the bin
    cov_miscen: np.ndarray | None = field(
        default=None, metadata={"component": True, "label": "miscentering"})     # McClintock+19 Sec. 5.2
    cov_ell: np.ndarray | None = field(
        default=None, metadata={"component": True, "label": "halo ellipticity"})  # Gruen+15 C^ell (halo_ell_cov.py)
    cov_clss: np.ndarray | None = field(
        default=None, metadata={"component": True, "label": "correlated LSS"})    # Gruen+15 C^corr (clss_cov.py)

    # ---- not a covariance component: population-mean profile, same units as sqrt(cov) ----
    DeltaSigma: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "R_hiMpc", np.array(self.R_hiMpc, dtype=float))
        object.__setattr__(self, "z_bin", tuple(float(v) for v in self.z_bin))
        object.__setattr__(self, "lambda_bin", tuple(float(v) for v in self.lambda_bin))
        n = self.R_hiMpc.size
        for f in fields(self):
            if f.metadata.get("component") and getattr(self, f.name) is not None:
                # own copy, so freezing it never freezes the caller's array
                object.__setattr__(self, f.name, np.array(getattr(self, f.name), dtype=float))
        for name, C in self.components().items():
            if C.shape != (n, n):
                raise ValueError(f"{name}: shape {C.shape}, expected ({n}, {n})")
            if not np.allclose(C, C.T):
                raise ValueError(f"{name}: not symmetric")
            if np.linalg.eigvalsh(C).min() < -1e-10 * np.abs(C).max():
                raise ValueError(f"{name}: not positive semi-definite")
            C.setflags(write=False)              # frozen in content, not just in name
        self.R_hiMpc.setflags(write=False)
        if self.DeltaSigma is not None:
            ds = np.array(self.DeltaSigma, dtype=float)
            if ds.shape != (n,):
                raise ValueError(f"DeltaSigma: shape {ds.shape}, expected ({n},)")
            ds.setflags(write=False)
            object.__setattr__(self, "DeltaSigma", ds)

    @classmethod
    def from_clens(cls, h, rp_Mpc, z_bin, lambda_bin, DeltaSigma=None, **covs):
        """Build from clens h-free quantities: rp [comoving Mpc], DeltaSigma [Msun/pc^2],
        covs [(Msun/pc^2)^2] passed as cov_stat=..., cov_cM=..., etc."""
        return cls(R_hiMpc=np.asarray(rp_Mpc) * h, z_bin=z_bin, lambda_bin=lambda_bin,
                   DeltaSigma=None if DeltaSigma is None else np.asarray(DeltaSigma) / h,
                   **{k: None if C is None else np.asarray(C) / h**2 for k, C in covs.items()})

    def components(self) -> dict[str, np.ndarray]:
        """Every component that was supplied, keyed by field name."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.metadata.get("component") and getattr(self, f.name) is not None}

    def merge(self, *others: "CovStackedDeltaSigma") -> "CovStackedDeltaSigma":
        """Combine the components computed by different calculators (IntrinsicCov, HaloEllCov, CLSSCov)
        for the same (z, lambda) bin and radial binning. A component present in several (cov_stat, which
        each calc_cov receives, or DeltaSigma) must agree; the first non-None one is kept."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        for o in others:
            if (o.R_hiMpc.shape != self.R_hiMpc.shape or not np.allclose(o.R_hiMpc, self.R_hiMpc, rtol=1e-10)
                    or o.z_bin != self.z_bin or o.lambda_bin != self.lambda_bin):
                raise ValueError("merge: different R binning, z_bin or lambda_bin")
            for f in fields(o):
                v = getattr(o, f.name)
                if f.name in ("R_hiMpc", "z_bin", "lambda_bin") or v is None:
                    continue
                if kw[f.name] is None:
                    kw[f.name] = v
                elif not np.allclose(kw[f.name], v, rtol=1e-8, atol=0):
                    raise ValueError(f"merge: {f.name} differs between the inputs")
        return type(self)(**kw)

    def matrix(self, name: str) -> np.ndarray:
        """One component by short name ('stat', 'cM', 'lambdaM', 'miscen', 'ell', 'clss'), or 'total'."""
        if name == "total":
            return self.total()
        C = self.components().get(f"cov_{name}")
        if C is None:
            short = [k.removeprefix("cov_") for k in self.components()]
            raise KeyError(f"'{name}' not in this covariance; available: {short + ['total']}")
        return C

    def total(self, *, include: list[str] | None = None) -> np.ndarray:
        """Sum of the components; `include` takes short names, e.g. ['stat', 'cM']."""
        if include is None:
            return sum(self.components().values())
        return sum(self.matrix(k) for k in include)

    def frac_diag(self) -> dict[str, np.ndarray]:
        """Each component's share of the total variance at each R."""
        d_tot = np.diag(self.total())
        return {k: np.diag(C) / d_tot for k, C in self.components().items()}

    @classmethod
    def label(cls, name: str) -> str:
        """Plot label for a field name ('cov_cM') or short name ('cM')."""
        name = name if name.startswith("cov_") else f"cov_{name}"
        return next(f.metadata["label"] for f in fields(cls) if f.name == name)

    # ---- I/O: one array per component, keyed by field name ----
    def save(self, path):
        extra = {} if self.DeltaSigma is None else {"DeltaSigma": self.DeltaSigma}
        np.savez(path, R_hiMpc=self.R_hiMpc,
                 z_bin=np.array(self.z_bin), lambda_bin=np.array(self.lambda_bin),
                 **self.components(), **extra)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            kw = {k: d[k] for k in d.files}
        kw["z_bin"] = tuple(kw["z_bin"])
        kw["lambda_bin"] = tuple(kw["lambda_bin"])
        return cls(**kw)


class IntrinsicCov(object):
    """
    Halo-to-halo covariance of the stacked DeltaSigma for one (z, lambda) bin.

    co, sr                : clens CosmoParameters and scaling relation (log-normal class, or HODRichness with hook enabled)
    zh_min, zh_max        : halo redshift bin (profiles evaluated at the midpoint)
    lam_min, lam_max      : richness bin
    f_mis, tau            : miscentering fraction and Gamma-distribution scale (f_mis=0 -> no miscentering)
    include_concentration : True -> log-normal c scatter at fixed M (sigma_lnc, n_c Gauss-Hermite nodes);
                            False -> every halo of mass M has the median c(M)
    cM                    : median c(M): 'duffy08' (Duffy+08 c200m(M,z)) or 'correa' (clens default)
    nM, n_mis, wcut       : mass nodes, miscentering-offset nodes, mass-weight cut (relative to peak)

    The slow clens setup (mass function, xi_mm) is done once in __init__; calc_cov can then be
    called for different radial binnings / N.
    """
    def __init__(self, co, sr, zh_min, zh_max, lam_min, lam_max, f_mis=0.0, tau=0.166,
                 include_concentration=True, cM="duffy08", sigma_lnc=0.35, n_c=7,
                 nM=40, n_mis=40, wcut=1e-3):
        if cM not in ("duffy08", "correa"):
            raise ValueError(f"unknown cM relation {cM!r}")
        self.co, self.sr = co, sr
        self.zh_min, self.zh_max = zh_min, zh_max
        self.zh = 0.5 * (zh_min + zh_max)
        self.lam_min, self.lam_max = lam_min, lam_max
        self.f_mis, self.tau = f_mis, tau
        self.include_concentration = include_concentration
        self.cM, self.sigma_lnc, self.n_c = cM, sigma_lnc, n_c
        self.nM, self.n_mis = nM, n_mis

        self.rho = cn.rho_crit_with_h * co.h**2 * co.OmegaM
        self.cf = CorrelationFunctions3D(z=self.zh, co=co, sr=sr, lambda_min=lam_min, lambda_max=lam_max)
        self.ximm = self.cf.xi_mm()

        # mass weights of clusters in the bin, on clens's fine grid -> coarse grid covering the support
        cf = self.cf
        w_fine = cf.dndlnM_arr * cf.rs.lnM_selection(cf.lnM_arr, self.zh)
        keep = w_fine > wcut * w_fine.max()
        self.lnM = np.linspace(cf.lnM_arr[keep][0], cf.lnM_arr[keep][-1], nM)
        w = np.interp(self.lnM, cf.lnM_arr, w_fine)
        self.w_M = w / w.sum()

        # miscentering offsets in units of R_lambda
        self.R_lambda = R_lambda_comoving(0.5 * (lam_min + lam_max), self.zh, co.h)
        self.x_mis = (np.arange(n_mis) + 0.5) / n_mis * 10 * tau
        px = self.x_mis / tau**2 * np.exp(-self.x_mis / tau)
        self.p_mis = px / px.sum()

        # concentration nodes: Gauss-Hermite in ln c around the median c(M)
        if include_concentration and sigma_lnc > 0 and n_c > 1:
            gh_x, gh_w = np.polynomial.hermite.hermgauss(n_c)
            self.dlnc, self.w_c = np.sqrt(2.) * sigma_lnc * gh_x, gh_w / np.sqrt(np.pi)
        else:
            self.dlnc, self.w_c = np.zeros(1), np.ones(1)

    def c_median(self, M):
        """median c200m at mass M [h-free Msun]"""
        if self.cM == "duffy08":
            return c_duffy08(M, self.zh, self.co.h, "200m")            # clens halos are M200m
        return self.cf.c200m_M200m_Correa(M)

    def calc_profiles(self, rp_edges):
        """
        binned DeltaSigma for every draw, on the (M, c, R_mis) grid:
            self.profiles : (nM, n_c, 1 + n_mis, n_rp); index 0 on the 3rd axis is the centred profile
            self.w_mis    : (1 + n_mis,) weights (1 - f_mis, f_mis * p_mis); w_M, w_c set in __init__
        """
        cf, rho = self.cf, self.rho
        R = np.exp(np.linspace(np.log(0.01), np.log(1.05 * rp_edges[-1]), 500))  # start small: DeltaSigma needs Sigma-bar(<R)
        n_k = 1 + (self.n_mis if self.f_mis > 0 else 0)
        P = np.zeros((len(self.lnM), len(self.dlnc), n_k, len(rp_edges) - 1))
        for i, lnm in enumerate(self.lnM):
            c_med = self.c_median(np.exp(lnm))
            xi2h = cf.xi_hm_2h(self.ximm, lnm)
            for j, dc in enumerate(self.dlnc):
                nfw = NFW(mass=np.exp(lnm), c=c_med * np.exp(dc), rho_mean=cf.rho_mean_z)
                xi = np.maximum(nfw.profile(cf.radius) / cf.rho_mean_z - 1., xi2h)
                rp, _, xirp, _ = DeltaSigmaR(cf.radius, xi, rp_max=150, rho=rho)
                Sig = rho * xirp * 1e-12                                 # Msun/pc^2
                # centred: integrate DeltaSigma from the clens grid (good inner boundary), then interpolate
                P[i, j, 0] = _bin_average(R, np.interp(R, rp, _delta_sigma(rp, Sig)), rp_edges)
                if n_k > 1:
                    Sig_mis = _ring_average(rp, Sig, R, self.x_mis * self.R_lambda)   # (n_mis, nR)
                    P[i, j, 1:] = _bin_average(R, _delta_sigma(R, Sig_mis), rp_edges)
        self.profiles = P
        self.w_mis = np.concatenate([[1. - self.f_mis], self.f_mis * self.p_mis]) if n_k > 1 else np.ones(1)
        return P

    def calc_cov(self, rp_edges, N, cov_stat):
        """
        rp_edges : radial bin edges [comoving Mpc, h-free], len n_rp+1 -- must match the clens binning,
                   i.e. exp(linspace(log(rp_min), log(rp_max), n_rp+1)) for cds.calc_cov(rp_min, rp_max, n_rp)
        N        : number of clusters in the stack (e.g. from ClusterCounts)
        cov_stat : clens Gaussian covariance, cds.cov_sum [(Msun/pc^2)^2, h-free]
        returns  : CovStackedDeltaSigma in h-units (also stored as self.result); cov_cM is None if
                   include_concentration=False, cov_miscen is None if f_mis=0

        Components follow the law of total covariance, nested M -> c -> R_mis:
            cov_lambdaM   = Cov_M[ E(DS | M) ]                           (mass spread in the richness bin)
            cov_cM        = E_M[ Cov_c( E(DS | M, c) ) ]                 (c scatter at fixed M)
            cov_miscen    = E_{M,c}[ Cov_mis( DS | M, c ) ]              (offset draws at fixed M, c)
        all divided by N; they sum exactly to the total. The split depends on this nesting order
        (e.g. the miscentering term is evaluated at each M, c), but the total does not.
        """
        P = self.calc_profiles(rp_edges)
        wM, wc, wk = self.w_M, self.w_c, self.w_mis
        mean_Mc = np.einsum("k,ijkr->ijr", wk, P)                     # E(DS | M, c)
        mean_M = np.einsum("j,ijr->ir", wc, mean_Mc)                   # E(DS | M)
        mean = wM @ mean_M

        dM = mean_M - mean
        dc = mean_Mc - mean_M[:, None, :]
        dk = P - mean_Mc[:, :, None, :]
        cov_M = np.einsum("i,ir,is->rs", wM, dM, dM) / N
        cov_c = np.einsum("i,j,ijr,ijs->rs", wM, wc, dc, dc) / N
        cov_k = np.einsum("i,j,k,ijkr,ijks->rs", wM, wc, wk, dk, dk) / N

        rp_edges = np.asarray(rp_edges)
        self.result = CovStackedDeltaSigma.from_clens(
            self.co.h, np.sqrt(rp_edges[:-1] * rp_edges[1:]),
            z_bin=(self.zh_min, self.zh_max), lambda_bin=(self.lam_min, self.lam_max), DeltaSigma=mean,
            cov_stat=cov_stat, cov_lambdaM=cov_M,
            cov_cM=cov_c if len(self.dlnc) > 1 else None,
            cov_miscen=cov_k if self.w_mis.size > 1 else None)
        return self.result
