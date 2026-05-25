"""
allpass.py — Polyphase allpass filter primitives.

Classes
-------
Filter          — (b, a) transfer-function container
AllPass         — Single 2nd-order allpass section:  H(z) = (c + z⁻²)/(1 + c·z⁻²)
AllPassChain    — Series of AllPass sections (one polyphase branch)
Delay           — Pure delay: z⁻ⁿ
LowPass         — Half-band LP:  (A0 + z⁻¹·A1) / 2
HighPass        — Half-band HP:  (A0 - z⁻¹·A1) / 2
Hilbert         — 90° phase-split pair
"""

from numpy import polysub, zeros, polymul, polyadd
import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def FilterAdd(f1, f2):
    den = polymul(f1.get_den(), f2.get_den())
    num = polyadd(polymul(f1.get_num(), f2.get_den()),
                  polymul(f1.get_den(), f2.get_num()))
    return Filter(num, den)


def FilterSub(f1, f2):
    den = polymul(f1.get_den(), f2.get_den())
    num = polysub(polymul(f1.get_num(), f2.get_den()),
                  polymul(f1.get_den(), f2.get_num()))
    return Filter(num, den)


def FilterMult(f1, f2):
    return Filter(polymul(f1.get_num(), f2.get_num()),
                  polymul(f1.get_den(), f2.get_den()))


def _mag_db(h):
    return 20 * np.log10(np.abs(h) + 1e-12)


def _freqz(b, a, fs, n=8192):
    """Return (freqs_hz, h) for a filter."""
    w, h = freqz(b, a, worN=n)
    return w * fs / (2 * np.pi), h


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

class Filter:
    """Rational transfer function container H(z) = B(z)/A(z)."""

    def __init__(self, b, a):
        self.b = b
        self.a = a

    def get_num(self):  return self.b
    def get_den(self):  return self.a

    def apply(self, x):
        return lfilter(self.b, self.a, x)

    def __repr__(self):
        return f"Filter(b={np.array2string(np.asarray(self.b), precision=6)},\n" \
               f"       a={np.array2string(np.asarray(self.a), precision=6)})"


class AllPass:
    """
    Single 2nd-order allpass section.

    For half-band LP/HP::

        H(z) = (c + z⁻²) / (1 + c·z⁻²)

    For Hilbert::

        H(z) = (-c + z⁻²) / (1 - c·z⁻²)
    """

    def __init__(self, b, a):
        self.b = b
        self.a = a

    def apply(self, x):
        return lfilter(self.b, self.a, x)

    def order(self):
        return len(self.a) - 1

    def get_transfer_function(self):
        return Filter(self.b, self.a)


class AllPassChain:
    """Series combination of AllPass sections (one polyphase branch)."""

    def __init__(self, filter_list=None):
        self.filters = filter_list if filter_list is not None else []

    def order(self):
        return self.filters[0].order() if self.filters else 0

    def append(self, filt):
        self.filters.append(filt)

    def apply_chain(self, x):
        for f in self.filters:
            x = f.apply(x)
        return x

    def get_transfer_function(self):
        num, den = 1, 1
        for f in self.filters:
            num = polymul(num, f.b)
            den = polymul(den, f.a)
        return Filter(num, den)

    def __repr__(self):
        coefs = [f.b[0] for f in self.filters]  # leading coef of each section
        return f"AllPassChain(coefs={coefs})"


class Delay:
    """Pure z⁻ⁿ delay."""

    def __init__(self, order):
        self._order = order

    def get_transfer_function(self):
        b = np.zeros(self._order + 1)
        a = np.zeros(self._order + 1)
        b[-1] = 1
        a[0] = 1
        return Filter(b, a)

    def get_num(self): return self.get_transfer_function().get_num()
    def get_den(self): return self.get_transfer_function().get_den()


# ---------------------------------------------------------------------------
# First-order allpass  H(z) = (b + z⁻¹) / (1 + b·z⁻¹)
# ---------------------------------------------------------------------------

class AllPassFirst:
    """
    Single first-order allpass section.

    H(z) = (b + z⁻¹) / (1 + b·z⁻¹)

    Note: the second-order sections used in LowPass/HighPass are the
    polyphase (z²) form of this same primitive — i.e., they are
    AllPassFirst evaluated at half the sample rate.  This class
    operates at full rate and does *not* decimate.
    """

    def __init__(self, b_coef):
        self.b_coef = float(b_coef)
        self._b = [b_coef, 1.0]
        self._a = [1.0, b_coef]

    def get_transfer_function(self):
        return Filter(list(self._b), list(self._a))

    def apply(self, x):
        return lfilter(self._b, self._a, x)


def _build_first_order_chain(b_coefs):
    """Return a list of AllPassFirst sections for the given coefficient array."""
    return [AllPassFirst(b) for b in b_coefs]


def _chain_tf(sections):
    """Multiply transfer functions of a list of sections."""
    tf = Filter([1.0], [1.0])
    for s in sections:
        tf = FilterMult(tf, s.get_transfer_function())
    return tf


class FirstOrderLP:
    """
    Single-rate low-pass filter built from first-order allpass sections.

    H_LP(z) = [1 + A(z)] / 2

    where  A(z) = ∏ (b_i + z⁻¹) / (1 + b_i·z⁻¹)

    This is a power-complementary LP/HP pair with a single allpass branch
    (no polyphase decomposition, no decimation).

    The cutoff is controlled by the coefficients; use
    :func:`pyhiir.design.first_order` to compute them from Hz.

    Args:
        b_coefs: Array of allpass coefficients (one per section).
    """

    def __init__(self, b_coefs):
        b_coefs = np.asarray(b_coefs, dtype=float)
        self.coefs = b_coefs
        self.branch_coefs = b_coefs          # single branch, no X/Y split
        self._sections = _build_first_order_chain(b_coefs)

    def get_transfer_function(self):
        A = _chain_tf(self._sections)
        one = Filter([1.0], [1.0])
        out = FilterAdd(one, A)              # 1 + A(z)
        return Filter(out.b, [2.0 * x for x in out.a])   # / 2

    def apply(self, x):
        tf = self.get_transfer_function()
        return lfilter(tf.b, tf.a, x)

    def info(self, fs=None):
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  FirstOrderLP   sections={len(self.coefs)}")
        if fs:
            print(f"  fs = {fs:.1f} Hz   (single-rate, no decimation)")
        print(f"{'='*58}")
        print(f"  Allpass branch coefs:")
        for k, b in enumerate(self.coefs):
            print(f"    [{k}]  b = {b:.12f}")
        print(f"  Transfer function b: {np.array2string(np.asarray(tf.b), precision=8)}")
        print(f"  Transfer function a: {np.array2string(np.asarray(tf.a), precision=8)}")

    def plot(self, fs=1.0, n_points=8192, title=None):
        title = title or f"FirstOrderLP  (sections={len(self.coefs)})"
        _plot_first_order(self, fs, n_points, title)


class FirstOrderHP:
    """
    Single-rate high-pass filter built from first-order allpass sections.

    H_HP(z) = [1 - A(z)] / 2

    Power-complementary partner of :class:`FirstOrderLP`.

    Args:
        b_coefs: Array of allpass coefficients (one per section).
    """

    def __init__(self, b_coefs):
        b_coefs = np.asarray(b_coefs, dtype=float)
        self.coefs = b_coefs
        self.branch_coefs = b_coefs
        self._sections = _build_first_order_chain(b_coefs)

    def get_transfer_function(self):
        A = _chain_tf(self._sections)
        one = Filter([1.0], [1.0])
        out = FilterSub(one, A)              # 1 - A(z)
        return Filter(out.b, [2.0 * x for x in out.a])   # / 2

    def apply(self, x):
        tf = self.get_transfer_function()
        return lfilter(tf.b, tf.a, x)

    def info(self, fs=None):
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  FirstOrderHP   sections={len(self.coefs)}")
        if fs:
            print(f"  fs = {fs:.1f} Hz   (single-rate, no decimation)")
        print(f"{'='*58}")
        print(f"  Allpass branch coefs:")
        for k, b in enumerate(self.coefs):
            print(f"    [{k}]  b = {b:.12f}")
        print(f"  Transfer function b: {np.array2string(np.asarray(tf.b), precision=8)}")
        print(f"  Transfer function a: {np.array2string(np.asarray(tf.a), precision=8)}")

    def plot(self, fs=1.0, n_points=8192, title=None):
        title = title or f"FirstOrderHP  (sections={len(self.coefs)})"
        _plot_first_order(self, fs, n_points, title)


def _plot_first_order(filt, fs, n_points, title):
    """Plot magnitude + phase for a FirstOrderLP or FirstOrderHP."""
    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    tf = filt.get_transfer_function()
    freqs, h = _freqz(tf.b, tf.a, fs, n_points)
    ax_m.plot(freqs, _mag_db(h), color='steelblue', lw=2.0, label='LP/HP')
    ax_p.plot(freqs, np.degrees(np.unwrap(np.angle(h))), color='steelblue', lw=2.0)
    ax_m.set_ylabel("Magnitude [dB]"); ax_m.set_ylim(-80, 5); ax_m.grid(True, alpha=0.3)
    ax_p.set_ylabel("Phase [°]"); ax_p.set_xlabel("Frequency [Hz]"); ax_p.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig


class FirstOrderBS:
    """
    Single-rate band-stop (block-band) filter from first-order allpass sections.

    H_BS(z) = H_LP(f_low) + H_HP(f_high)
            = [1 + A_low(z)] / 2  +  [1 − A_high(z)] / 2
            = 1  +  [A_low(z) − A_high(z)] / 2

    where:
        A_low(z)  = ∏ (b_i + z⁻¹) / (1 + b_i·z⁻¹)  tuned to f_low
        A_high(z) = ∏ (b_j + z⁻¹) / (1 + b_j·z⁻¹)  tuned to f_high

    This passes all frequencies below f_low and above f_high, and
    attenuates the band [f_low, f_high].  The depth and selectivity
    depend on the number of sections and how far apart f_low and f_high are.

    Note: DC gain = 1.0,  Nyquist gain = 1.0.

    Args:
        b_low:  Allpass coefficients for the LP branch (from :func:`first_order`
                called with f_pass = f_low).
        b_high: Allpass coefficients for the HP branch (from :func:`first_order`
                called with f_pass = f_high).

    Example::

        from pyhiir.design import first_order
        from pyhiir.allpass import FirstOrderBS

        # Block the quarter-band [fs/8, fs/4] = [187, 375] Hz at fs=1500
        b_low  = first_order(fs=1500, f_pass=187, n=3)
        b_high = first_order(fs=1500, f_pass=375, n=3)
        bs = FirstOrderBS(b_low, b_high)
    """

    def __init__(self, b_low, b_high):
        b_low  = np.asarray(b_low,  dtype=float)
        b_high = np.asarray(b_high, dtype=float)
        self.coefs_low  = b_low
        self.coefs_high = b_high
        # coefs: concatenated for compatibility with status bar / show_single
        self.coefs      = np.concatenate([b_low, b_high])
        self._lp = FirstOrderLP(b_low)
        self._hp = FirstOrderHP(b_high)

    def get_transfer_function(self):
        """H_BS(z) = H_LP(z) + H_HP(z)."""
        tf_lp = self._lp.get_transfer_function()
        tf_hp = self._hp.get_transfer_function()
        return FilterAdd(Filter(tf_lp.b, tf_lp.a),
                         Filter(tf_hp.b, tf_hp.a))

    def apply(self, x):
        tf = self.get_transfer_function()
        return lfilter(tf.b, tf.a, x)

    def info(self, fs=None):
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  FirstOrderBS   LP sections={len(self.coefs_low)}  HP sections={len(self.coefs_high)}")
        if fs:
            print(f"  fs = {fs:.1f} Hz   (single-rate, no decimation)")
        print(f"{'='*58}")
        print(f"  LP branch coefs (f_low):  {self.coefs_low}")
        print(f"  HP branch coefs (f_high): {self.coefs_high}")
        print(f"  b = {np.array2string(np.asarray(tf.b), precision=8)}")
        print(f"  a = {np.array2string(np.asarray(tf.a), precision=8)}")


# ---------------------------------------------------------------------------
# Butterworth filter  (scipy-based, expressed as b/a)
# ---------------------------------------------------------------------------

class ButterworthFilter:
    """
    Classical Butterworth LP or HP filter via bilinear transform (scipy).

    Not allpass-based; included for comparison in the 1st-order filter tab.

    Args:
        b, a:   Numerator / denominator coefficients (from scipy.signal.butter).
        ftype:  'LP' or 'HP'.
        order:  Filter order.
        f_pass: Cutoff frequency [Hz].
    """

    def __init__(self, b, a, ftype='LP', order=1, f_pass=None):
        self.b_coef = np.asarray(b, dtype=float)
        self.a_coef = np.asarray(a, dtype=float)
        self.ftype  = ftype
        self.coefs  = self.b_coef           # display compatibility
        self._order = order
        self.f_pass = f_pass

    def get_transfer_function(self):
        return Filter(list(self.b_coef), list(self.a_coef))

    def apply(self, x):
        return lfilter(self.b_coef, self.a_coef, x)


# ---------------------------------------------------------------------------
# DC Blocker  H(z) = (1 - z⁻¹) / (1 - r·z⁻¹)
# ---------------------------------------------------------------------------

class DCBlocker:
    """
    Classic DC-blocking highpass filter.

    H(z) = (1 − z⁻¹) / (1 − r·z⁻¹)

    Zero at z=1 (DC = 0 Hz), pole at z=r (just below DC).

    The coefficient r controls the −3 dB frequency:
        r ≈ 1 − 2π·f_c/fs    (first-order approximation)
        r = exp(−2π·f_c/fs)  (more accurate)

    Args:
        r:  Pole position (0 < r < 1, typically > 0.99 for audio).
    """

    def __init__(self, r):
        r = float(r)
        if not (0 < r < 1):
            raise ValueError(f"r must be in (0, 1), got {r}")
        self.r     = r
        self.coefs = np.array([r])          # display compatibility
        self._b    = [1.0, -1.0]
        self._a    = [1.0, -r]

    @property
    def f_c(self):
        """Approximate −3 dB frequency [normalised, 0–0.5]."""
        import numpy as _np
        return _np.arccos(2 * self.r / (1 + self.r**2)) / (2 * _np.pi)

    def get_transfer_function(self):
        return Filter(list(self._b), list(self._a))

    def apply(self, x):
        return lfilter(self._b, self._a, x)


# ---------------------------------------------------------------------------
# Half-band filters  (polyphase, 2nd-order = 1st-order in z² domain)
# ---------------------------------------------------------------------------

def _build_halfband_branches(coef):
    """Split flat coef array into (AllPassChain_X, AllPassChain_Y)."""
    bi = [AllPass([c, 0, 1], [1, 0, c]) for c in coef[0::2]]
    by = [AllPass([c, 0, 1], [1, 0, c]) for c in coef[1::2]]
    return AllPassChain(bi), AllPassChain(by)


class LowPass:
    """
    Half-band low-pass decimator.

    H_LP(z) = [A0(z²) + z⁻¹·A1(z²)] / 2

    Args:
        coef: Polyphase allpass coefficients (from :func:`pyhiir.design.halfband`).
              Even-indexed → Branch X.  Odd-indexed → Branch Y.
    """

    def __init__(self, coef):
        coef = np.asarray(coef)
        self.coefs = coef
        self.branch_x_coefs = coef[0::2]
        self.branch_y_coefs = coef[1::2]
        self.bi, self.by = _build_halfband_branches(coef)

    def get_transfer_function(self):
        bi_tf = self.bi.get_transfer_function()
        by_tf = FilterMult(Delay(1).get_transfer_function(),
                           self.by.get_transfer_function())
        out = FilterAdd(bi_tf, by_tf)
        return Filter(out.get_num(), 2 * out.get_den())

    def get_num(self): return self.get_transfer_function().get_num()
    def get_den(self): return self.get_transfer_function().get_den()

    def apply(self, x, decimate=True):
        """Filter signal, optionally decimate by 2."""
        tf = self.get_transfer_function()
        y = lfilter(tf.b, tf.a, x)
        return y[::2] if decimate else y

    def info(self, fs=None):
        """Print branch coefficients and transfer function."""
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  LowPass  (half-band)   order={len(self.coefs)}")
        if fs:
            print(f"  fs = {fs:.1f} Hz  →  {fs/2:.1f} Hz  (decimation ×2)")
        print(f"{'='*58}")
        print(f"  Branch X coefs  [even idx]:")
        for k, c in enumerate(self.branch_x_coefs):
            print(f"    [{k}]  {c:.10f}")
        print(f"  Branch Y coefs  [odd idx]:")
        for k, c in enumerate(self.branch_y_coefs):
            print(f"    [{k}]  {c:.10f}")
        print(f"  Transfer function b:\n    {np.array2string(np.asarray(tf.b), precision=8)}")
        print(f"  Transfer function a:\n    {np.array2string(np.asarray(tf.a), precision=8)}")

    def plot(self, fs=1.0, n_points=8192, title=None):
        """
        Plot magnitude response of combined filter and both branches.

        Args:
            fs: Sample rate [Hz] for x-axis labelling (use 1.0 for normalised).
        """
        title = title or f"LowPass  (order={len(self.coefs)})"
        _plot_halfband(self, fs, n_points, title)


class HighPass:
    """
    Half-band high-pass decimator.

    H_HP(z) = [A0(z²) − z⁻¹·A1(z²)] / 2

    Args:
        coef: Polyphase allpass coefficients (same array as for LowPass).
    """

    def __init__(self, coef):
        coef = np.asarray(coef)
        self.coefs = coef
        self.branch_x_coefs = coef[0::2]
        self.branch_y_coefs = coef[1::2]
        self.bi, self.by = _build_halfband_branches(coef)

    def get_transfer_function(self):
        bi_tf = self.bi.get_transfer_function()
        by_tf = FilterMult(Delay(1).get_transfer_function(),
                           self.by.get_transfer_function())
        out = FilterSub(bi_tf, by_tf)
        return Filter(out.get_num(), 2 * out.get_den())

    def get_num(self): return self.get_transfer_function().get_num()
    def get_den(self): return self.get_transfer_function().get_den()

    def apply(self, x, decimate=True):
        """Filter signal, optionally decimate by 2."""
        tf = self.get_transfer_function()
        y = lfilter(tf.b, tf.a, x)
        return y[::2] if decimate else y

    def info(self, fs=None):
        """Print branch coefficients and transfer function."""
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  HighPass  (half-band)  order={len(self.coefs)}")
        if fs:
            print(f"  fs = {fs:.1f} Hz  →  {fs/2:.1f} Hz  (decimation ×2)")
        print(f"{'='*58}")
        print(f"  Branch X coefs  [even idx]:")
        for k, c in enumerate(self.branch_x_coefs):
            print(f"    [{k}]  {c:.10f}")
        print(f"  Branch Y coefs  [odd idx]:")
        for k, c in enumerate(self.branch_y_coefs):
            print(f"    [{k}]  {c:.10f}")
        print(f"  Transfer function b:\n    {np.array2string(np.asarray(tf.b), precision=8)}")
        print(f"  Transfer function a:\n    {np.array2string(np.asarray(tf.a), precision=8)}")

    def plot(self, fs=1.0, n_points=8192, title=None):
        """
        Plot magnitude response of combined filter and both branches.

        Args:
            fs: Sample rate [Hz] for x-axis labelling (use 1.0 for normalised).
        """
        title = title or f"HighPass  (order={len(self.coefs)})"
        _plot_halfband(self, fs, n_points, title)


# ---------------------------------------------------------------------------
# Plotting helper (shared by LP and HP)
# ---------------------------------------------------------------------------

def _plot_halfband(filt, fs, n_points, title):
    """Two-panel plot: magnitude + phase, with branch overlays."""
    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    tf   = filt.get_transfer_function()
    bx_tf = filt.bi.get_transfer_function()
    by_tf = filt.by.get_transfer_function()

    xlabel = "Frequency [Hz]" if fs != 1.0 else "Normalised freq  [×π rad/sample]"

    specs = [
        (tf,    "Combined", "steelblue",  2.0),
        (bx_tf, "Branch X", "tomato",     1.2),
        (by_tf, "Branch Y", "seagreen",   1.2),
    ]

    for f_obj, lbl, col, lw in specs:
        freqs, h = _freqz(f_obj.b, f_obj.a, fs, n_points)
        ax_m.plot(freqs, _mag_db(h),        label=lbl, color=col, lw=lw)
        ax_p.plot(freqs, np.degrees(np.angle(h)), label=lbl, color=col, lw=lw)

    ax_m.set_ylabel("Magnitude [dB]")
    ax_m.set_ylim(-80, 5)
    ax_m.grid(True, alpha=0.3)
    ax_m.legend()

    ax_p.set_ylabel("Phase [°]")
    ax_p.set_xlabel(xlabel)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()

    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Hilbert pair
# ---------------------------------------------------------------------------

class Hilbert:
    """
    Hilbert-transform polyphase pair.

    Returns two allpass chains with ≈90° phase difference across the band.
    """

    def __init__(self, coef):
        coef = np.asarray(coef)
        self.coefs = coef
        self.branch_x_coefs = coef[0::2]
        self.branch_y_coefs = coef[1::2]
        bi = [AllPass([c, 0, -1], [1, 0, -c]) for c in coef[0::2]]
        by = [AllPass([c, 0, -1], [1, 0, -c]) for c in coef[1::2]]
        self.bi = AllPassChain(bi)
        self.by = AllPassChain(by)

    def get_transfer_function(self):
        bx_tf = self.bi.get_transfer_function()
        by_tf = FilterMult(Delay(1).get_transfer_function(),
                           self.by.get_transfer_function())
        s = Filter([1], [np.sqrt(2)])
        bx = FilterMult(s, FilterAdd(by_tf, bx_tf))
        by = FilterMult(s, FilterSub(by_tf, bx_tf))
        return bx, by

    def apply(self, x):
        hi, hq = self.get_transfer_function()
        hx = lfilter(hi.b, hi.a, x)
        hy = lfilter(hq.b, hq.a, x)
        return hx + 1j * hy

    def info(self):
        """Print branch coefficients."""
        print(f"\n{'='*58}")
        print(f"  Hilbert pair   order={len(self.coefs)}")
        print(f"{'='*58}")
        print(f"  Branch X coefs:")
        for k, c in enumerate(self.branch_x_coefs):
            print(f"    [{k}]  {c:.10f}")
        print(f"  Branch Y coefs:")
        for k, c in enumerate(self.branch_y_coefs):
            print(f"    [{k}]  {c:.10f}")



# ---------------------------------------------------------------------------
# Quarter-band band-stop  (polyphase, single-rate)
# ---------------------------------------------------------------------------

class QuarterBandBS:
    """
    Wide band-stop filter using the halfband-of-halfband z→z² symmetry.

    H_BS(z) = H_LP(z²)

    The z→z² substitution makes H_LP periodic with half its original period,
    creating a naturally symmetric response around fs/4:

        |H_BS(f)| = |H_LP(2f)|

        Pass:  [0, ≈fs/8]    and  [≈3fs/8, fs/2]  (Q1 and Q4)
        Stop:  [≈fs/8, ≈3fs/8]                      (Q2 and Q3)
        Null:  exactly fs/4  (always −∞ dB)
        -3 dB: exactly fs/8  and  3fs/8  (halfband identity)

    Order controls the transition steepness and stopband depth.
    The -3 dB frequencies (fs/8 and 3fs/8) are fixed by the structure.

    Args:
        coefs: Allpass coefficients from a standard halfband LP design.
    """

    def __init__(self, coefs):
        coefs = np.asarray(coefs, dtype=float)
        self.coefs          = coefs
        self.branch_x_coefs = coefs[0::2]
        self.branch_y_coefs = coefs[1::2]
        self._lp = LowPass(coefs)

    @staticmethod
    def _upsample(arr):
        """Insert a zero between every coefficient: z → z²."""
        arr = np.asarray(arr, dtype=float)
        out = np.zeros(2 * len(arr) - 1)
        out[::2] = arr
        return out

    def get_transfer_function(self):
        """H_BS(z) = H_LP(z²)  — symmetric wide stop at [fs/8, 3fs/8]."""
        tf = self._lp.get_transfer_function()
        b2 = self._upsample(tf.b)
        a2 = self._upsample(tf.a)
        return Filter(list(b2), list(a2))

    def apply(self, x):
        tf = self.get_transfer_function()
        return lfilter(tf.b, tf.a, x)

    def info(self, fs=None):
        tf = self.get_transfer_function()
        print(f"\n{'='*58}")
        print(f"  QuarterBandBS  H_LP(z\u00b2)  order={len(self.coefs)}")
        if fs:
            print(f"  fs={fs:.1f} Hz   stopband \u2248[{fs/8:.1f}\u2013{3*fs/8:.1f}] Hz"
                  f"  (passes Q1+Q4)")
        print(f"{'='*58}")
        print(f"  coefs = {self.coefs}")
        b = np.asarray(tf.b); a = np.asarray(tf.a)
        print(f"  b = {np.array2string(b, precision=8)}")
        print(f"  a = {np.array2string(a, precision=8)}")


# ---------------------------------------------------------------------------
# ASCII transfer-function renderer
# ---------------------------------------------------------------------------

def _section_ascii(c, prec=8):
    """
    Return (top, bar, bot) strings for one allpass section::

        ( c + z⁻² )
        ─────────────
        (1 + c·z⁻²)
    """
    cs  = f"{c:.{prec}f}"
    top = f"({cs} + z\u207b\u00b2)"
    bot = f"(1 + {cs}\u00b7z\u207b\u00b2)"
    w   = max(len(top), len(bot))
    return top.center(w), "\u2500" * w, bot.center(w)


def _branch_ascii(coefs, prec=8):
    """
    Render a branch (product of allpass sections) as three lines::

        (c0 + z⁻²)   (c1 + z⁻²)
        ──────────── × ────────────
        (1+c0·z⁻²)   (1+c1·z⁻²)
    """
    if len(coefs) == 0:
        return "  1   (identity)"
    sections = [_section_ascii(c, prec) for c in coefs]
    sep_top = "   "
    sep_mid = " \u00d7 "   # ' × '
    sep_bot = "   "
    top = sep_top.join(s[0] for s in sections)
    mid = sep_mid.join(s[1] for s in sections)
    bot = sep_bot.join(s[2] for s in sections)
    return f"  {top}\n  {mid}\n  {bot}"


def ascii_tf(filt, fs=None, prec=8):
    """
    Return an ASCII string showing the transfer function as a product of
    allpass sections, per branch.

    Example output for a LowPass(order=4)::

        LowPass   order=4   fs=1500.0 Hz → 750.0 Hz

        Branch X:  A₀(z²)
          (0.07986643 + z⁻²)   (0.54532365 + z⁻²)
          ────────────────────── × ──────────────────────
          (1 + 0.07986643·z⁻²)  (1 + 0.54532365·z⁻²)

        Branch Y:  A₁(z²)  [with z⁻¹ delay]
          (0.28382934 + z⁻²)   (0.83441189 + z⁻²)
          ────────────────────── × ──────────────────────
          (1 + 0.28382934·z⁻²)  (1 + 0.83441189·z⁻²)

        Combined:  H(z) = [ A₀(z²) + z⁻¹·A₁(z²) ] / 2

    Args:
        filt:  LowPass, HighPass, or Hilbert instance.
        fs:    Sample rate [Hz] for header annotation (optional).
        prec:  Decimal digits for coefficients.

    Returns:
        str
    """
    lines = []
    ftype = type(filt).__name__
    hdr   = f"{ftype}   order={len(filt.coefs)}"
    if fs is not None:
        hdr += f"   fs={fs:.1f} Hz → {fs/2:.1f} Hz"
    lines += [hdr, ""]

    is_hilbert = isinstance(filt, Hilbert)

    # Branch X
    lines.append("Branch X:  A\u2080(z\u00b2)")
    lines.append(_branch_ascii(filt.branch_x_coefs, prec))
    lines.append("")

    # Branch Y
    delay_note = "" if is_hilbert else "  [with z\u207b\u00b9 delay]"
    lines.append(f"Branch Y:  A\u2081(z\u00b2){delay_note}")
    lines.append(_branch_ascii(filt.branch_y_coefs, prec))
    lines.append("")

    # Combiner
    if isinstance(filt, LowPass):
        lines.append("Combined:  H(z) = [ A\u2080(z\u00b2) + z\u207b\u00b9\u00b7A\u2081(z\u00b2) ] / 2")
    elif isinstance(filt, HighPass):
        lines.append("Combined:  H(z) = [ A\u2080(z\u00b2) \u2212 z\u207b\u00b9\u00b7A\u2081(z\u00b2) ] / 2")
    elif is_hilbert:
        lines.append("I output:  I(z) = [ A\u2080(z\u00b2) + z\u207b\u00b9\u00b7A\u2081(z\u00b2) ] / \u221a2")
        lines.append("Q output:  Q(z) = [ A\u2081(z\u00b2)\u00b7z\u207b\u00b9 \u2212 A\u2080(z\u00b2) ] / \u221a2")

    return "\n".join(lines)
