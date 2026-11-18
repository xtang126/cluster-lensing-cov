#!/usr/bin/env python
"""
Analytic NFW lensing profile, ported from CLensPy (src/clenspy/halo/nfw.py,
https://github.com/estevesjh/CLensPy) so cov_clens does not depend on the package.

Port changes (everything else is verbatim):
  * `scalar_array_output` copied in from clenspy/utils/decorators.py;
  * `rho_ref` is required (CLensPy defaults to its fiducial cosmology, which is not here);
  * the pyccl comparison demo under __main__ is dropped;
  * ADDED, truncated at r200 (the upstream projected quantities are untruncated):
        sigma_truncated(R)   Sigma(R) of rho_NFW * Theta(r200 - r)   (Takada & Jain 2003, MNRAS 340, 580, Eq. 27)
        mass_2d(R, truncated=True)   projected mass inside a cylinder of radius R (closed form, derived below)

Units: h-free, Msun, Mpc (comoving if rho_ref is comoving, as in clens).

---- CLensPy licence ----
MIT License

Copyright (c) 2025 J.H. Esteves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.special import sici


def scalar_array_output(method):
    """(from clenspy.utils.decorators) Return a Python scalar if the first positional argument
    was scalar-like and the result is a single number; otherwise return the array unchanged."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        scalar_in = np.ndim(args[0]) == 0
        if scalar_in:
            if isinstance(result, np.ndarray):
                if result.size == 1:
                    return result.reshape(()).item()
                return result
            return float(result)
        return result
    return wrapper


class NfwProfile:
    r"""
    Analytical NFW lensing profile for a single halo or a vector of halos.

    NOTE: **the mass definition is carried by ``rho_ref``**, the reference
    density that closes :math:`M_{200} = 200\,\rho_{\rm ref}\,
    \frac{4}{3}\pi r_{200}^3`. This class fixes only the overdensity 200;
    which density it is measured against is the caller's choice. Pass the
    comoving mean matter density and ``m200`` means M_200m; pass the
    critical density and it means M_200c. Mixing the two is a ~30% mass
    error, so whoever supplies ``rho_ref`` owns that decision.

    NOTE: this class carries no cosmology. That one density is the only
    cosmological input an NFW profile needs; everything else is geometry.
    Pass the density, not a cosmology.

    NOTE: all quantities are h-free absolute units -- mass in Msun,
    lengths in Mpc, densities in Msun/Mpc^3, wavenumbers in 1/Mpc.

    The 3D density profile is

    .. math::
        \rho(r) = \frac{\rho_s}{x (1 + x)^2}, \qquad x = \frac{r}{r_s}

    where the scale radius :math:`r_s = r_{200} / c_{200}` and the
    characteristic density :math:`\rho_s` are fixed by requiring the profile
    to enclose :math:`M_{200}` within :math:`r_{200}`,

    .. math::
        \rho_s = \frac{M_{200}}
        {4\pi r_s^3 \left[\ln(1 + c_{200}) - c_{200} / (1 + c_{200})\right]}.

    Parameters
    ----------
    m200 : float, array-like
        Halo mass [Msun] within r200, w.r.t. 200x ``rho_ref``. Can be
        scalar or array.
    c200 : float, array-like
        Concentration c_200 = r_200 / r_s (dimensionless). Can be scalar
        or array.
    rho_ref : float, optional
        Reference density [Msun/Mpc^3] defining the overdensity, and with
        it the mass definition. Defaults to `mean_matter_density()` -- the
        comoving mean matter density at z=0 of the fiducial cosmology,
        making the default definition M_200m. (cov_clens port: required.)

    Notes
    -----
    All methods are vectorized -- pass whole ``r``/``R``/``k`` arrays,
    never loop element-by-element. Observed shapes: with a scalar halo,
    scalar in gives a Python float and ``(N,)`` in gives ``(N,)`` out;
    with a vector of halos ``(n_halo,)``, array input gives
    ``(n_halo, N)`` and scalar input gives ``(n_halo, 1)`` (`fourier`
    squeezes the k axis, giving ``(n_halo,)``).
    """

    def __init__(
        self,
        m200: np.ndarray | float,
        c200: np.ndarray | float = 4.0,
        rho_ref: float | None = None,
    ) -> None:
        # Broadcast shapes for mass and concentration
        m200, c200 = np.broadcast_arrays(m200, c200)
        self.m200 = m200
        self.c200 = c200
        if rho_ref is None:   # cov_clens port: no fiducial cosmology here -- pass clens's rho_mean
            raise ValueError("rho_ref is required (e.g. CorrelationFunctions3D.rho_mean_z for M200m)")
        self.rho_ref = float(rho_ref)

        # Calculate r200 and rs
        self.r200 = self._calculateAtR200(self.m200)  # (n_halo,)
        self.rs = self.r200 / self.c200  # (n_halo,)
        self.rho_s = self._calculateRhos(self.m200, self.c200)  # (n_halo,)

    def _calculateAtR200(self, m200: np.ndarray | float) -> np.ndarray | float:
        """Calculate r200 [Mpc] for given m200 [Msun]."""
        m200 = np.asarray(m200)
        return (3 * m200 / (4 * np.pi * 200 * self.rho_ref)) ** (1.0 / 3.0)

    def _calculateRhos(
        self, m200: np.ndarray | float, c200: np.ndarray | float
    ) -> np.ndarray | float:
        """Calculate NFW scale density rho_s [Msun/Mpc^3]."""
        c200 = np.asarray(c200)
        rho_s = m200 / (4 * np.pi * self.rs**3 * (np.log(1 + c200) - c200 / (1 + c200)))
        return rho_s

    @scalar_array_output
    def density(self, r: np.ndarray | float) -> np.ndarray | float:
        r"""
        Calculate 3D density profile for NFW.

        .. math::
            \rho(r) = \frac{\rho_s}{x (1 + x)^2}, \qquad x = \frac{r}{r_s}

        Parameters
        ----------
        r : float or np.ndarray
            Radius [Mpc]. Can be scalar or array.
        Returns
        -------
        rho : np.ndarray
            Density [Msun/Mpc^3], shape = broadcast(n_halo, n_r)
        """
        r = np.atleast_1d(r)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        x = r / rs
        return rho_s / (x * (1 + x) ** 2)

    @scalar_array_output
    def fourier(
        self, k: np.ndarray | float, truncated: bool = True
    ) -> np.ndarray | float:
        r"""
        Fourier transform of the NFW density profile, :math:`\tilde\rho(k)`.

        .. math::
            \tilde\rho(k) \equiv \int d^3r\, \rho_{\rm NFW}(r)\,
            e^{i \mathbf{k} \cdot \mathbf{r}}

        evaluated with the closed-form expression (see e.g. pyccl, or eq. 81
        of Cooray & Sheth 2002), with :math:`x \equiv k r_s`. For the profile
        truncated at :math:`r_{200} = c_{200} r_s`,

        .. math::
            \tilde\rho(k) = \frac{M}{\ln(1+c) - c/(1+c)} \Big\{
            \sin(x)\left[\mathrm{Si}\big((1+c)x\big) - \mathrm{Si}(x)\right]
            + \cos(x)\left[\mathrm{Ci}\big((1+c)x\big) - \mathrm{Ci}(x)\right]
            - \frac{\sin(cx)}{(1+c)x} \Big\},

        and for the untruncated (infinite-extent) profile,

        .. math::
            \tilde\rho(k) = \frac{M}{\ln(1+c) - c/(1+c)} \left\{
            \sin(x)\left[\frac{\pi}{2} - \mathrm{Si}(x)\right]
            - \cos(x)\, \mathrm{Ci}(x) \right\},

        where :math:`c \equiv c_{200}` and Si, Ci are the sine and cosine
        integrals.

        NOTE: this returns :math:`\tilde\rho(k)`, carrying units of mass
        [Msun] and going to :math:`M` (not :math:`1`) as :math:`k \to 0` --
        it is *not* the dimensionless mass-normalized :math:`u(k\mid M)
        \equiv \tilde\rho(k)/M` that most halo-model formulas mean by "the
        profile's Fourier transform", including this package's own
        ``LensingProfile.fourier_profile`` (``clenspy/lensing/profile.py``),
        which adds this method's output to a 2-halo term already divided by
        ``m200`` -- a units mismatch, not yet fixed there. This method's
        convention matches `pyccl`'s ``HaloProfileNFW.fourier`` exactly
        (verified numerically, both truncated and not); divide by
        ``self.m200`` at the call site to get :math:`u(k\mid M)`.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber array [1/Mpc].
        truncated : bool, optional
            If True, use truncated Fourier transform (default: True).

        Returns
        -------
        rho_tilde : np.ndarray
            :math:`\tilde\rho(k)` [Msun], *not* mass-normalized.
            Shape: (n_halo, n_k)
        """
        k_in = k
        k = np.atleast_1d(np.asarray(k, dtype=float))
        m200, c200, rs = np.broadcast_arrays(
            np.atleast_1d(self.m200), np.atleast_1d(self.c200),
            np.atleast_1d(self.rs),
        )
        # NOTE: explicit (n_halo, n_k) layout. The previous spelling put a
        # second ``[..., None]`` on P1, which was already (n_halo, n_k), so
        # an array ``m200`` broadcast to (n_halo, n_k, n_k) -- silently, and
        # only for array mass. Scalar mass was correct, which is why it
        # survived: every test passed a scalar.
        x = rs[:, None] * k[None, :]
        norm = np.log(1 + c200) - c200 / (1 + c200)      # (n_halo,)
        P1 = (m200 / norm)[:, None]
        Si2, Ci2 = sici(x)
        if truncated:
            Si1, Ci1 = sici((1 + c200)[:, None] * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c200[:, None] * x) / ((1 + c200)[:, None] * x)
            prof = P1 * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1 * P2

        # squeeze on the *input* shapes, not the broadcast ones
        if np.ndim(self.m200) == 0:
            prof = prof[0]
        if np.ndim(k_in) == 0:
            prof = prof[..., 0]
        return prof

    @scalar_array_output
    def sigma(self, R: np.ndarray | float) -> np.ndarray | float:
        r"""
        Projected surface density Σ(R) for NFW, in [Msun/Mpc^2].

        .. math::
            \Sigma(R) = 2 r_s \rho_s\, f(x), \qquad x = \frac{R}{r_s}

        with the piecewise kernel (Wright & Brainerd 2000)

        .. math::
            f(x) =
            \begin{cases}
            \dfrac{1}{x^2 - 1}\left[1 - \dfrac{2}{\sqrt{1 - x^2}}\,
            \mathrm{arctanh}\sqrt{\dfrac{1-x}{1+x}}\right], & x < 1 \\[2mm]
            \dfrac{1}{3}, & x = 1 \\[2mm]
            \dfrac{1}{x^2 - 1}\left[1 - \dfrac{2}{\sqrt{x^2 - 1}}\,
            \arctan\sqrt{\dfrac{x-1}{x+1}}\right], & x > 1
            \end{cases}

        Parameters
        ----------
        R : float or np.ndarray
            Projected radius [Mpc].

        Returns
        -------
        sigma : np.ndarray
            Projected surface density, shape = broadcast(n_halo, n_R)
        """
        R = np.atleast_1d(R)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        Rs = R / rs
        sigma = 2 * rs * rho_s * self._fNfw(Rs)
        return sigma

    @scalar_array_output
    def mean_sigma(self, R: np.ndarray | float) -> np.ndarray | float:
        r"""
        Mean interior surface density :math:`\bar\Sigma(<R)`, in [Msun/Mpc^2].

        .. math::
            \bar\Sigma(<R) = \frac{2}{R^2}\int_0^R \Sigma(R')\, R'\, dR'
            = 2 r_s \rho_s\, \bar{g}(x), \qquad x = R / r_s

        with the closed form of `_gbarNfw`. Equal to
        :math:`\Sigma(R) + \Delta\Sigma(R)` analytically, but evaluated
        directly: forming it as that sum cancels catastrophically at small
        :math:`x` (see `_gbarNfw`).

        Parameters
        ----------
        R : float or np.ndarray
            Projected radius [Mpc].

        Returns
        -------
        mean_sigma : np.ndarray
            Mean interior surface density, shape = broadcast(n_halo, n_R)
        """
        R = np.atleast_1d(R)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        return 2 * rs * rho_s * self._gbarNfw(R / rs)

    @scalar_array_output
    def deltasigma(self, R: np.ndarray | float) -> np.ndarray | float:
        r"""
        Excess surface density ΔΣ(R) for NFW, in [Msun/Mpc^2].

        .. math::
            \Delta\Sigma(R) \equiv \bar\Sigma(<R) - \Sigma(R)
            = r_s \rho_s\, g(x), \qquad x = \frac{R}{r_s}

        with the piecewise kernel (Wright & Brainerd 2000)

        .. math::
            g(x) =
            \begin{cases}
            \dfrac{8\,\mathrm{arctanh}\sqrt{\frac{1-x}{1+x}}}{x^2\sqrt{1-x^2}}
            + \dfrac{4}{x^2}\ln\dfrac{x}{2} - \dfrac{2}{x^2-1}
            + \dfrac{4\,\mathrm{arctanh}\sqrt{\frac{1-x}{1+x}}}{(x^2-1)\sqrt{1-x^2}},
            & x < 1 \\[3mm]
            \dfrac{10}{3} + 4\ln\dfrac{1}{2}, & x = 1 \\[3mm]
            \dfrac{8\arctan\sqrt{\frac{x-1}{x+1}}}{x^2\sqrt{x^2-1}}
            + \dfrac{4}{x^2}\ln\dfrac{x}{2} - \dfrac{2}{x^2-1}
            + \dfrac{4\arctan\sqrt{\frac{x-1}{x+1}}}{(x^2-1)^{3/2}}, & x > 1
            \end{cases}

        Parameters
        ----------
        R : float or np.ndarray
            Projected radius [Mpc].

        Returns
        -------
        deltasigma : np.ndarray
            Excess surface density, shape = broadcast(n_halo, n_R)
        """
        R = np.atleast_1d(R)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        x = R / rs
        deltasigma = rs * rho_s * self._gNfw(x)
        return deltasigma

    # Taylor coefficients of f(1+d) and g(1+d) about the x=1 kink of the
    # piecewise closed forms, where the direct expressions lose ~1e-16/|x-1|
    # to 0/0 cancellation. Exact values (sympy, series in w^2=(1-x)/(1+x)):
    #   f: 1/3, -2/5, 13/35, -20/63, 61/231, -94/429, 1181/6435,
    #      -1896/12155, 6223/46189
    #   g: p_n + (-1)^(n+1) 4(n+1) ln2, p_n = 10/3, -88/15, 296/35,
    #      -3508/315, 1373/99, -49930/3003, 9601/495, -186556/8415,
    #      11521457/461890
    # With |x-1| <= _SERIES_WINDOW the truncation error is |c_9| d^9 ~ 1e-19
    # while the direct branch outside the window is accurate to <~2e-14.
    _F_SERIES = np.array([
        1 / 3, -2 / 5, 13 / 35, -20 / 63, 61 / 231, -94 / 429,
        1181 / 6435, -1896 / 12155, 6223 / 46189,
    ])
    _G_SERIES = np.array([
        10 / 3 - 4 * np.log(2), -88 / 15 + 8 * np.log(2),
        296 / 35 - 12 * np.log(2), -3508 / 315 + 16 * np.log(2),
        1373 / 99 - 20 * np.log(2), -49930 / 3003 + 24 * np.log(2),
        9601 / 495 - 28 * np.log(2), -186556 / 8415 + 32 * np.log(2),
        11521457 / 461890 - 36 * np.log(2),
    ])
    _SERIES_WINDOW = 1e-2

    #: Below this x, the closed form for gbar cancels; use the series.
    #: Chosen where the two meet -- both are ~5e-11 there, against mpmath.
    _GBAR_SMALL_X = 3e-3

    @classmethod
    def _gbarNfw(cls, x):
        r"""Mean interior projected NFW kernel
        :math:`\bar{g}(x) = \bar\Sigma(<x) / (2 r_s \rho_s)`.

        .. math::
            \bar{g}(x) = \frac{2}{x^2}\left[\ln\frac{x}{2}
            + \begin{cases}
              \operatorname{arccosh}(1/x)/\sqrt{1 - x^2}, & x < 1\\
              1, & x = 1\\
              \arccos(1/x)/\sqrt{x^2 - 1}, & x > 1
              \end{cases}\right]

        NOTE: evaluated from this closed form rather than reconstructed as
        :math:`f + g/2`. The bracket is a difference of two terms that both
        behave like :math:`\ln(2/x)` while their sum is
        :math:`O(x^2\ln x)`, so the reconstruction loses every digit below
        :math:`x \sim 10^{-6}` -- `_gNfw` there returns values that are
        orders of magnitude wrong and eventually negative. The same
        cancellation eventually reaches this form too, hence the series
        branch below `_GBAR_SMALL_X`:

        .. math::
            \bar{g}(x) = \ln\frac{2}{x} - \frac{1}{2}
            + x^2\left[\frac{3}{4}\ln\frac{2}{x} - \frac{7}{16}\right]
            + O(x^4\ln x)

        Both branches agree with mpmath to <= 5e-11 at the switch.
        """
        x = np.array(x, dtype=float)
        out = np.empty_like(x)

        tiny = x < cls._GBAR_SMALL_X
        if np.any(tiny):
            L = np.log(2.0 / x[tiny])
            out[tiny] = L - 0.5 + x[tiny] ** 2 * (0.75 * L - 7.0 / 16.0)

        rest = ~tiny
        lo = rest & (x < 1.0 - 1e-8)
        hi = rest & (x > 1.0 + 1e-8)
        eq = rest & ~(lo | hi)
        xl, xh = x[lo], x[hi]
        out[lo] = (2.0 / xl**2) * (
            np.log(xl / 2.0) + np.arccosh(1.0 / xl) / np.sqrt(1.0 - xl * xl)
        )
        out[hi] = (2.0 / xh**2) * (
            np.log(xh / 2.0) + np.arccos(1.0 / xh) / np.sqrt(xh * xh - 1.0)
        )
        out[eq] = 2.0 * (1.0 + np.log(0.5))
        return out

    @classmethod
    def _fNfw(cls, x):
        """Projected NFW profile kernel f(x)."""
        x = np.array(x, dtype=float)
        result = np.zeros_like(x)
        eps = cls._SERIES_WINDOW
        mask1 = x < 1 - eps
        mask3 = x > 1 + eps
        mask2 = ~(mask1 | mask3)
        x1 = x[mask1]
        x3 = x[mask3]
        # For x < 1. Written with arccosh(1/x) rather than the equivalent
        # 2 arctanh(sqrt((1-x)/(1+x))): the arctanh argument rounds to
        # exactly 1 once x drops below ~1e-17, giving arctanh(1) = inf,
        # and the miscentering integrand samples x -> 0 whenever the
        # azimuthal ring passes through the halo centre (R = R_mis).
        # arccosh(1/x) stays accurate down to the 1/x overflow.
        result[mask1] = (
            1.0
            / (x1**2 - 1.0)
            * (1 - np.arccosh(1.0 / x1) / np.sqrt(1 - x1**2))
        )
        # For |x - 1| <= eps: Taylor series (direct form is 0/0 at x=1)
        result[mask2] = np.polynomial.polynomial.polyval(
            x[mask2] - 1.0, cls._F_SERIES
        )
        # For x > 1
        result[mask3] = (
            1.0
            / (x3**2 - 1.0)
            * (1 - 2 / np.sqrt(x3**2 - 1) * np.arctan(np.sqrt((x3 - 1) / (x3 + 1))))
        )
        return result

    @classmethod
    def _gNfw(cls, x, eps=None):
        """Mean enclosed projected NFW kernel g(x)."""
        if eps is None:
            eps = cls._SERIES_WINDOW
        x = np.array(x, dtype=float)
        res = np.zeros_like(x)
        mask_l = x < 1 - eps
        mask_g_ = x > 1 + eps
        # near x = 1: Taylor series (direct form is 0/0 at x=1)
        mask_c = ~(mask_l | mask_g_)
        res[mask_c] = np.polynomial.polynomial.polyval(
            x[mask_c] - 1.0, cls._G_SERIES
        )
        # Small x: the 1/x^2 terms below are individually ~ln(2/x)/x^2 and
        # cancel down to O(1), so the direct form loses every digit by
        # x ~ 1e-6 and turns negative by 1e-9. Use the limit instead,
        #     g(x) = 1 - x^2 [ (3/2) ln(2/x) - 13/8 ] + O(x^4 ln x),
        # which follows from g = 2(gbar - f) and the expansions of both.
        tiny = mask_l & (x < cls._GBAR_SMALL_X)
        if np.any(tiny):
            L = np.log(2.0 / x[tiny])
            res[tiny] = 1.0 - x[tiny] ** 2 * (1.5 * L - 13.0 / 8.0)
        mask_l = mask_l & ~tiny

        sqrt1mx2 = np.sqrt(1.0 - x[mask_l] ** 2)
        # arccosh(1/x) == 2 arctanh(sqrt((1-x)/(1+x))), but stays finite
        # once the arctanh argument would round to exactly 1 (see _fNfw).
        acosh = np.arccosh(1.0 / x[mask_l])
        term1 = 4.0 * acosh / (x[mask_l] ** 2 * sqrt1mx2)
        term2 = 4.0 / x[mask_l] ** 2 * np.log(x[mask_l] / 2.0)
        term3 = -2.0 / (x[mask_l] ** 2 - 1.0)
        term4 = 2.0 * acosh / ((x[mask_l] ** 2 - 1.0) * sqrt1mx2)
        res[mask_l] = term1 + term2 + term3 + term4

        # x > 1
        mask_g = mask_g_
        sqrtx2m1 = np.sqrt(x[mask_g] ** 2 - 1.0)
        atan = np.arctan(sqrtx2m1 / (1.0 + x[mask_g]))
        term1 = 8.0 * atan / (x[mask_g] ** 2 * sqrtx2m1)
        term2 = 4.0 / x[mask_g] ** 2 * np.log(x[mask_g] / 2.0)
        term3 = -2.0 / (x[mask_g] ** 2 - 1.0)
        term4 = 4.0 * atan / ((x[mask_g] ** 2 - 1.0) ** 1.5)
        res[mask_g] = term1 + term2 + term3 + term4

        return res

    # ------------------------------------------------------------------------------------------
    # cov_clens additions: NFW truncated at r200 (rho = 0 for r > r200), total mass exactly m200.
    # Output shape is (*halo_shape, *R.shape) for R of ANY shape, so a vector of haloes can be
    # evaluated on a whole N-d grid in one call.
    # X = R / rs, c = c200.  Exact near X = 1 (no 0/0) and at X -> 0 (no cancellation) as noted.
    # ------------------------------------------------------------------------------------------
    @scalar_array_output
    def sigma_truncated(self, R: np.ndarray | float) -> np.ndarray | float:
        r"""
        Projected surface density of the NFW truncated at r200, [Msun/Mpc^2]; 0 for R >= r200.

        .. math::
            \Sigma(R) = 2 r_s \rho_s F(X), \qquad X = R / r_s \le c

        (Takada & Jain 2003, Eq. 27):

        .. math::
            F = -\frac{\sqrt{c^2-X^2}}{(1-X^2)(1+c)} + \frac{1}{(1-X^2)^{3/2}}
                \operatorname{arccosh}\frac{X^2+c}{X(1+c)}  \quad (X<1),

        and the analytic continuation arccosh -> -arccos, (1-X^2)^{3/2} -> (X^2-1)^{3/2} for X > 1.
        Both pieces are 0/0 at X = 1; inside |X-1| < 1e-3 the value is linearly interpolated from
        X = 1 -+ 1e-3 (F is smooth there; error ~1e-7 relative).
        """
        R = np.atleast_1d(R)
        rs, rho_s, c, _ = self._halo_axes(R)
        X = R / rs
        return 2 * rs * rho_s * self._fTrunc(X, np.broadcast_to(c, X.shape))

    @scalar_array_output
    def mass_2d(self, R: np.ndarray | float, truncated: bool = True) -> np.ndarray | float:
        r"""
        Projected mass inside a cylinder of radius R about the halo centre, [Msun].

        Untruncated: :math:`\pi R^2 \bar\Sigma(<R)` (upstream `mean_sigma`, diverges as ln R).
        Truncated at r200 (-> m200 for R >= r200): every shell of radius r > R loses the fraction
        :math:`\sqrt{1 - R^2/r^2}` of its mass outside the cylinder, so

        .. math::
            M_{2D}(<R) = M_{200} - 4\pi\rho_s r_s^3\, J(X), \qquad
            J = \int_X^c \frac{\sqrt{t^2 - X^2}}{(1+t)^2} dt,

        which integrates (by parts, then t = X cosh(theta), tau = tanh(theta/2)) to

        .. math::
            J = -\frac{\sqrt{c^2-X^2}}{1+c} + \operatorname{arccosh}\frac{c}{X} - K,\quad
            K = \frac{2}{\sqrt{1-X^2}} \operatorname{arctanh}\Big(\tau_c\sqrt{\tfrac{1-X}{1+X}}\Big)
            \ (X<1),\quad
            \frac{2}{\sqrt{X^2-1}} \arctan\Big(\tau_c\sqrt{\tfrac{X-1}{X+1}}\Big)\ (X>1),

        :math:`\tau_c = \sqrt{(c-X)/(c+X)}`, K = tau_c at X = 1. For X < 1e-3 the two logarithms in J
        cancel, so there the untruncated mass minus the (tiny) part beyond r200 is used instead:

        .. math::
            M_{2D} = 2\pi\rho_s r_s^3 X^2 \Big[\bar g(X) - \ln\frac{1+c}{c} + \frac{1}{1+c}\Big]
                     + O(X^4).
        """
        R = np.atleast_1d(R)
        rs, rho_s, c, m200 = self._halo_axes(R)
        X = R / rs
        if not truncated:
            return np.pi * R**2 * 2 * rs * rho_s * self._gbarNfw(X)
        c = np.broadcast_to(c, X.shape)
        return np.where(X >= c, m200, 4 * np.pi * rs**3 * rho_s * self._m2dTrunc(X, c))

    def _halo_axes(self, R):
        """rs, rho_s, c200, m200 reshaped to (*halo_shape, 1, ..., 1) with one trailing axis per axis
        of R, so the truncated methods return (*halo_shape, *R.shape) for R of any shape
        (identical to the upstream [..., None] convention when R is 1-d)."""
        pad = (1,) * R.ndim
        return tuple(np.reshape(a, np.shape(a) + pad) for a in (self.rs, self.rho_s, self.c200, self.m200))

    _TRUNC_WINDOW = 1e-3
    _M2D_SMALL_X = 1e-3

    @classmethod
    def _fTrunc_direct(cls, X, c):
        X = np.asarray(X, float); c = np.asarray(c, float)
        out = np.zeros_like(X)
        lo = X < 1
        hi = (X > 1) & (X < c)
        for m, sgn in ((lo, 1.), (hi, -1.)):
            x, cc = X[m], c[m]
            arg = (x * x + cc) / (x * (1 + cc))
            ang = np.arccosh(arg) if sgn > 0 else np.arccos(np.clip(arg, -1., 1.))
            out[m] = (-np.sqrt(cc * cc - x * x) / ((1 - x * x) * (1 + cc))
                      + sgn * ang / np.abs(1 - x * x) ** 1.5)
        return out

    @classmethod
    def _fTrunc(cls, X, c):
        X = np.asarray(X, float); c = np.asarray(c, float)
        out = cls._fTrunc_direct(X, c)
        e = cls._TRUNC_WINDOW
        near = (np.abs(X - 1) < e) & (X < c)
        if np.any(near):
            cn_ = c[near]
            fl = cls._fTrunc_direct(np.full(cn_.shape, 1 - e), cn_)
            fh = cls._fTrunc_direct(np.full(cn_.shape, 1 + e), cn_)
            out[near] = fl + (fh - fl) * (X[near] - (1 - e)) / (2 * e)
        return out

    @classmethod
    def _m2dTrunc(cls, X, c):
        """M_2D(<R) / (4 pi rho_s rs^3) for the truncated NFW, valid for X < c."""
        X = np.asarray(X, float); c = np.asarray(c, float)
        norm = np.log(1 + c) - c / (1 + c)
        out = np.array(norm, dtype=float, copy=True)          # X >= c (caller masks), and fallback
        tiny = X < cls._M2D_SMALL_X
        if np.any(tiny):
            x, cc = X[tiny], c[tiny]
            out[tiny] = 0.5 * x * x * (cls._gbarNfw(x) - np.log((1 + cc) / cc) + 1 / (1 + cc))
        m = ~tiny & (X < c)
        x, cc = X[m], c[m]
        tau = np.sqrt((cc - x) / (cc + x))
        K = np.empty_like(x)
        lo, hi = x < 1 - 1e-12, x > 1 + 1e-12
        eq = ~(lo | hi)
        K[lo] = 2 / np.sqrt(1 - x[lo]**2) * np.arctanh(tau[lo] * np.sqrt((1 - x[lo]) / (1 + x[lo])))
        K[hi] = 2 / np.sqrt(x[hi]**2 - 1) * np.arctan(tau[hi] * np.sqrt((x[hi] - 1) / (x[hi] + 1)))
        K[eq] = tau[eq]
        J = -np.sqrt(cc * cc - x * x) / (1 + cc) + np.arccosh(cc / x) - K
        out[m] = norm[m] - J
        return out


__all__ = ["NfwProfile"]
