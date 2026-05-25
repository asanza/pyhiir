"""
design.py — frequency specs → polyphase allpass coefficients.

Usage::

    from pyhiir.design import halfband
    coefs = halfband(fs=1500, f_pass=300, order=4)
    lp = LowPass(coefs)
"""

import numpy as np
from .hiir import hiir as _Hiir

_designer = None


def _get_designer():
    global _designer
    if _designer is None:
        _designer = _Hiir()
    return _designer


def tbw_from_freqs(fs, f_pass, f_stop=None):
    """
    Compute normalized transition bandwidth for a half-band decimator.

    For a symmetric half-band: f_stop = fs/2 - f_pass.
    TBW = (f_stop - f_pass) / fs

    Args:
        fs:     Input sample rate [Hz]
        f_pass: Passband edge [Hz]   (must be < fs/4)
        f_stop: Stopband edge [Hz]   (default: fs/2 - f_pass)

    Returns:
        tbw (float): normalised transition bandwidth > 0
    """
    if f_stop is None:
        f_stop = fs / 2.0 - f_pass
    tbw = (f_stop - f_pass) / fs
    if tbw <= 0:
        raise ValueError(
            f"f_pass={f_pass:.2f} Hz must be < fs/4={fs/4:.2f} Hz "
            f"for a half-band decimator (got TBW={tbw:.5f})."
        )
    return float(tbw)


def halfband(fs, f_pass, order=None, attenuation_db=60.0, f_stop=None):
    """
    Design polyphase allpass coefficients for a half-band decimator.

    Even-indexed coefficients → Branch X (bi).
    Odd-indexed coefficients  → Branch Y (by).

    Args:
        fs:             Input sample rate [Hz]
        f_pass:         Passband edge [Hz]  (absolute, at this stage's fs)
        order:          Number of allpass sections. None → auto from attenuation_db.
        attenuation_db: Stopband attenuation [dB]  (used when order=None).
        f_stop:         Stopband edge [Hz]  (default: fs/2 - f_pass).

    Returns:
        coefs: np.ndarray, shape (order,)

    Example::

        coefs = halfband(fs=1500, f_pass=300, order=4)
        lp = LowPass(coefs)
        lp.plot(fs=1500)
    """
    tbw = tbw_from_freqs(fs, f_pass, f_stop)
    d = _get_designer()
    if order is not None:
        return d.compute_coefs_order_tbw(int(order), tbw)
    return d.compute_coefs(float(attenuation_db), tbw)


def first_order(fs, f_pass, n=1):
    """
    Design first-order allpass coefficients for single-rate LP/HP filters.

    Each section:  H_i(z) = (b_i + z⁻¹) / (1 + b_i·z⁻¹)
    Combined:      A(z)   = ∏ H_i(z)
    LP:            H_LP(z) = [1 + A(z)] / 2
    HP:            H_HP(z) = [1 - A(z)] / 2

    For n=1 the coefficient is exact (bilinear transform of a 1st-order
    Butterworth):
        b = (1 − tan(π·f_pass/fs)) / (1 + tan(π·f_pass/fs))

    For n>1 the coefficients are computed by minimising the maximum
    stopband attenuation of the LP filter using scipy.optimize.

    Note: Unlike :func:`halfband`, this produces **non-decimating** filters.
    The equivalent decimating filter would use the same coefficients but in
    the z² (second-order) polyphase form, which is what :class:`LowPass` and
    :class:`HighPass` already do.

    Args:
        fs:     Sample rate [Hz].
        f_pass: Target −3 dB frequency [Hz]  (for n=1; approximated for n>1).
        n:      Number of first-order allpass sections  (1 = exact, >1 optimised).

    Returns:
        coefs: np.ndarray, shape (n,)  — allpass coefficients b_0 … b_{n-1}.

    Example::

        from pyhiir.design import first_order
        from pyhiir.allpass import FirstOrderLP, FirstOrderHP

        b = first_order(fs=1500, f_pass=300, n=3)
        lp = FirstOrderLP(b)
        hp = FirstOrderHP(b)
    """
    import numpy as np
    omega_c = np.tan(np.pi * f_pass / fs)

    if n == 1:
        b = (1.0 - omega_c) / (1.0 + omega_c)
        return np.array([b])

    # For n > 1: minimise max stopband ripple via Chebyshev-like equiripple search.
    # The stopband starts at fs/2 - f_pass (symmetric about fs/4).
    from scipy.optimize import minimize
    from scipy.signal import freqz

    f_stop = fs / 2.0 - f_pass
    w_stop_start = 2.0 * np.pi * f_stop / fs      # stopband start (rad/sample)
    n_pts = 512

    def _lp_response(b_arr):
        """Magnitude-squared of LP = (1 + ∏H_i) / 2  at n_pts frequencies."""
        # Build combined transfer function numerically
        num = np.array([1.0])
        den = np.array([1.0])
        for b in b_arr:
            num = np.convolve(num, [b, 1.0])
            den = np.convolve(den, [1.0, b])
        # LP = (den + num) / (2 * den)
        lp_b = (num + den) / 2.0    # numerator of LP (before normalising den)
        lp_a = den
        return lp_b, lp_a

    def objective(b_arr):
        lp_b, lp_a = _lp_response(b_arr)
        w = np.linspace(w_stop_start, np.pi, n_pts, endpoint=True)
        _, h = freqz(lp_b, lp_a, worN=w)
        return np.max(20.0 * np.log10(np.abs(h) + 1e-15))   # max stopband dB → minimise

    # Initial guess: n identical bilinear sections tuned to f_pass
    b0 = (1.0 - omega_c) / (1.0 + omega_c)
    x0 = np.full(n, b0)
    bounds = [(-0.9999, 0.9999)] * n

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 2000})

    return np.sort(res.x)[::-1]   # sort descending (convention: sharpest first)


def first_order_bs(fs, f_low, f_high, n=1):
    """
    Design first-order allpass coefficients for a band-stop filter.

    Returns two coefficient arrays:
        b_low  — LP branch, −3 dB at f_low  (passes [0, f_low])
        b_high — HP branch, −3 dB at f_high (passes [f_high, fs/2])

    The combined filter:
        H_BS(z) = H_LP(b_low) + H_HP(b_high)
                = 1 + [A_low(z) − A_high(z)] / 2

    blocks the band [f_low, f_high] and passes everything outside it.
    For the "quarter-band block" set f_low ≈ fs/8, f_high ≈ fs/4.

    Args:
        fs:     Sample rate [Hz].
        f_low:  Lower stopband edge (LP −3 dB point) [Hz].
        f_high: Upper stopband edge (HP −3 dB point) [Hz].
        n:      Number of first-order allpass sections per branch.

    Returns:
        (b_low, b_high): tuple of np.ndarray, each shape (n,).

    Example::

        from pyhiir.design import first_order_bs
        from pyhiir.allpass import FirstOrderBS

        b_low, b_high = first_order_bs(fs=1500, f_low=187, f_high=375, n=3)
        bs = FirstOrderBS(b_low, b_high)
    """
    if f_low >= f_high:
        raise ValueError(f"f_low ({f_low}) must be < f_high ({f_high})")
    if f_high >= fs / 2:
        raise ValueError(f"f_high ({f_high}) must be < fs/2 ({fs/2})")
    b_low  = first_order(fs, f_low,  n=n)
    b_high = first_order(fs, f_high, n=n)
    return b_low, b_high


def butterworth(fs, f_pass, order=1, ftype='LP'):
    """
    Design a Butterworth LP or HP filter (scipy bilinear transform).

    Args:
        fs:     Sample rate [Hz].
        f_pass: −3 dB cutoff frequency [Hz].
        order:  Filter order (any positive integer).
        ftype:  'LP' or 'HP'.

    Returns:
        ButterworthFilter instance.
    """
    from scipy.signal import butter
    from .allpass import ButterworthFilter
    Wn  = f_pass / (fs / 2.0)
    Wn  = float(np.clip(Wn, 1e-6, 1 - 1e-6))
    btype = 'low' if ftype == 'LP' else 'high'
    b, a  = butter(order, Wn, btype=btype)
    return ButterworthFilter(b, a, ftype=ftype, order=order, f_pass=f_pass)


def dc_blocker(fs, f_c):
    """
    Design a DC-blocking filter H(z) = (1 − z⁻¹) / (1 − r·z⁻¹).

    The pole radius is computed as r = exp(−2π·f_c/fs).

    Args:
        fs:  Sample rate [Hz].
        f_c: Approximate −3 dB cutoff frequency [Hz]  (typically 1–20 Hz).

    Returns:
        DCBlocker instance.
    """
    from .allpass import DCBlocker
    r = float(np.exp(-2.0 * np.pi * f_c / fs))
    return DCBlocker(r)


def quarterband_bs(fs, order=None, attenuation_db=60.0, tbw=0.1):
    """
    Auto quarter-band polyphase band-stop filter.

    Blocks [≈fs/8, ≈fs/4] and passes [0, fs/8] and [fs/4, fs/2].
    Power complement of the quarter-band bandpass.

    Internally designs two polyphase allpass half-band filters:
        LP: transition around fs/8  (narrow LP, passes [0, fs/8])
        HP: transition around fs/4  (wide HP, passes [fs/4, fs/2])

    Args:
        fs:             Input sample rate [Hz].
        order:          Allpass order per branch (None = auto from attenuation_db).
        attenuation_db: Stopband attenuation [dB] when order is None.
        tbw:            Relative transition bandwidth (0–0.25, default 0.1).

    Returns:
        QuarterBandBS instance.

    Example::

        from pyhiir.design import quarterband_bs
        bs = quarterband_bs(fs=1500, order=4)
    """
    from .allpass import QuarterBandBS

    # Both stages use the same half-band design (symmetric around fs/4).
    # The z→z² substitution in get_transfer_function() makes the HP
    # act at half-rate, giving an effective cutoff at fs/8.
    coefs = halfband(fs, fs * 0.20, order=order, attenuation_db=attenuation_db)
    return QuarterBandBS(coefs)
