"""
chain.py — Multi-stage decimation filter chain.

Usage::

    from pyhiir.chain import DecimatorChain

    chain = DecimatorChain(fs=1500)
    chain.add_lp(f_pass=300, order=4)   # 1500 → 750 Hz
    chain.add_lp(f_pass=150, order=2)   # 750  → 375 Hz
    chain.info()
    chain.plot()
    y = chain.apply(x)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

from .allpass import LowPass, HighPass, FirstOrderLP, FirstOrderHP, _mag_db, _freqz
from .design import halfband, first_order


# ---------------------------------------------------------------------------
# StageSpec
# ---------------------------------------------------------------------------

class StageSpec:
    """A single decimation stage (filter + ×2 downsampling)."""

    def __init__(self, idx, ftype, input_fs, f_pass, coefs, filt):
        self.stage_idx   = idx
        self.filter_type = ftype        # 'LP' | 'HP' | 'BS'
        self.input_fs    = input_fs
        self.output_fs   = input_fs if ftype == 'BS' else input_fs / 2.0
        self.f_pass      = f_pass
        self.coefs       = coefs
        self._filter     = filt
        self._is_first_order = False
        self._decimates  = (ftype != 'BS')

    # --- branch coefficient access ---

    @property
    def branch_x_coefs(self):
        """Branch X coefficients (even-indexed for polyphase, all for 1st-order)."""
        if self._is_first_order:
            return self.coefs           # single branch
        return self.coefs[0::2]

    @property
    def branch_y_coefs(self):
        """Branch Y coefficients (odd-indexed for polyphase, empty for 1st-order)."""
        if self._is_first_order:
            return np.array([])         # no Y branch
        return self.coefs[1::2]

    # --- transfer function ---

    def get_transfer_function(self):
        """Return combined Filter(b, a)."""
        return self._filter.get_transfer_function()

    # --- signal processing ---

    def apply(self, x):
        """Filter and optionally decimate by 2."""
        tf = self.get_transfer_function()
        y  = lfilter(tf.b, tf.a, x)
        return y if not self._decimates else y[::2]

    # --- introspection ---

    def info(self):
        """Print full coefficient and transfer-function summary."""
        tf = self.get_transfer_function()
        sep = '=' * 60
        print(f"\n{sep}")
        print(f"  Stage {self.stage_idx}: {self.filter_type}   "
              f"{self.input_fs:.1f} Hz → {self.output_fs:.1f} Hz")
        print(sep)
        print(f"  f_pass  : {self.f_pass:.4f} Hz")
        print(f"  Sections: {len(self.coefs)}  "
              f"(X: {len(self.branch_x_coefs)},  Y: {len(self.branch_y_coefs)})")
        print(f"\n  Branch X coefficients  (even-indexed):")
        for k, c in enumerate(self.branch_x_coefs):
            print(f"    [{k}]  c = {c:.12f}")
        print(f"\n  Branch Y coefficients  (odd-indexed):")
        for k, c in enumerate(self.branch_y_coefs):
            print(f"    [{k}]  c = {c:.12f}")
        print(f"\n  Combined transfer function:")
        print(f"    b = {np.array2string(np.asarray(tf.b), precision=8, separator=', ')}")
        print(f"    a = {np.array2string(np.asarray(tf.a), precision=8, separator=', ')}")

    def plot(self, n_points=8192):
        """Plot magnitude response for this stage (combined + branches)."""
        fs = self.input_fs
        title = (f"Stage {self.stage_idx}: {self.filter_type}  "
                 f"{fs:.0f}→{self.output_fs:.0f} Hz,  f_pass={self.f_pass:.1f} Hz")

        fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(title, fontsize=12, fontweight='bold')

        tf    = self.get_transfer_function()
        bx_tf = self._filter.bi.get_transfer_function()
        by_tf = self._filter.by.get_transfer_function()

        for f_obj, lbl, col, lw in [
            (tf,    "Combined", "steelblue", 2.0),
            (bx_tf, "Branch X", "tomato",    1.3),
            (by_tf, "Branch Y", "seagreen",  1.3),
        ]:
            freqs, h = _freqz(f_obj.b, f_obj.a, fs, n_points)
            ax_m.plot(freqs, _mag_db(h),               label=lbl, color=col, lw=lw)
            ax_p.plot(freqs, np.degrees(np.angle(h)),  label=lbl, color=col, lw=lw)

        ax_m.axvline(self.f_pass, color='orange', ls=':', lw=1.2, label='f_pass')
        ax_m.set_ylabel("Magnitude [dB]");  ax_m.set_ylim(-80, 5)
        ax_m.grid(True, alpha=0.3);         ax_m.legend(fontsize=8)
        ax_p.set_ylabel("Phase [°]");       ax_p.set_xlabel("Frequency [Hz]")
        ax_p.grid(True, alpha=0.3);         ax_p.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        return fig


# ---------------------------------------------------------------------------
# DecimatorChain
# ---------------------------------------------------------------------------

class DecimatorChain:
    """
    Multi-stage decimation filter chain.

    Each stage applies a half-band LP or HP filter and decimates by 2.
    Frequencies are always specified in absolute Hz at the *current* stage's
    sample rate (tracked automatically as stages are added).

    Example::

        chain = DecimatorChain(fs=1500)
        chain.add_lp(f_pass=300, order=4)   # 1500 → 750 Hz
        chain.add_lp(f_pass=150, order=2)   # 750  → 375 Hz
        chain.info()
        chain.plot()
        y = chain.apply(x)
    """

    def __init__(self, fs):
        self.fs     = float(fs)
        self.stages = []

    # --- current fs tracking ---

    @property
    def output_fs(self):
        """Sample rate at the output of the last stage."""
        return self.stages[-1].output_fs if self.stages else self.fs

    def fs_at(self, stage_idx):
        """Sample rate at the output of stage `stage_idx`."""
        return self.stages[stage_idx].output_fs

    def _current_input_fs(self):
        return self.stages[-1].output_fs if self.stages else self.fs

    # --- adding stages ---

    def add_lp(self, f_pass, order=None, attenuation_db=60.0, f_stop=None):
        """
        Append a half-band LP decimation stage.

        Args:
            f_pass:         Passband edge [Hz] at this stage's input fs.
            order:          Number of allpass sections (None = auto).
            attenuation_db: Stopband attenuation [dB]  (used when order=None).
            f_stop:         Stopband edge [Hz]  (default: input_fs/2 - f_pass).
        """
        fs    = self._current_input_fs()
        coefs = halfband(fs, f_pass, order=order,
                         attenuation_db=attenuation_db, f_stop=f_stop)
        filt  = LowPass(coefs)
        self.stages.append(StageSpec(len(self.stages), 'LP', fs, f_pass, coefs, filt))
        return self  # chainable

    def add_hp(self, f_pass, order=None, attenuation_db=60.0, f_stop=None):
        """
        Append a half-band HP decimation stage.

        Args:
            f_pass:         Passband edge [Hz] at this stage's input fs.
            order:          Number of allpass sections (None = auto).
            attenuation_db: Stopband attenuation [dB]  (used when order=None).
            f_stop:         Stopband edge [Hz]  (default: input_fs/2 - f_pass).
        """
        fs    = self._current_input_fs()
        coefs = halfband(fs, f_pass, order=order,
                         attenuation_db=attenuation_db, f_stop=f_stop)
        filt  = HighPass(coefs)
        self.stages.append(StageSpec(len(self.stages), 'HP', fs, f_pass, coefs, filt))
        return self  # chainable

    def add_lp1(self, f_pass, n=1):
        """
        Append a **first-order allpass** low-pass decimation stage.

        Uses H_LP(z) = [1 + A(z)] / 2  where A(z) = ∏(bᵢ+z⁻¹)/(1+bᵢz⁻¹),
        then downsamples by 2.

        Unlike :meth:`add_lp` (which runs the polyphase branches at fs/2),
        this filter runs at the *full* input rate and is thus less efficient
        but simpler — each coefficient is a scalar real number.

        f_pass is the −3 dB frequency for n=1, or the optimised stopband
        edge for n>1  (see :func:`pyhiir.design.first_order`).

        Args:
            f_pass: Target passband/−3 dB frequency [Hz].
            n:      Number of first-order allpass sections (1 = one coefficient).
        """
        fs    = self._current_input_fs()
        coefs = first_order(fs, f_pass, n=n)
        filt  = FirstOrderLP(coefs)
        self.stages.append(StageSpec(len(self.stages), 'LP1', fs, f_pass, coefs, filt))
        return self

    def add_hp1(self, f_pass, n=1):
        """
        Append a **first-order allpass** high-pass decimation stage.

        Uses H_HP(z) = [1 − A(z)] / 2, then downsamples by 2.

        Power-complementary partner of :meth:`add_lp1`.

        Args:
            f_pass: Target passband/−3 dB frequency [Hz].
            n:      Number of first-order allpass sections.
        """
        fs    = self._current_input_fs()
        coefs = first_order(fs, f_pass, n=n)
        filt  = FirstOrderHP(coefs)
        self.stages.append(StageSpec(len(self.stages), 'HP1', fs, f_pass, coefs, filt))
        return self

    def add_quarterband_bs(self, order=None, attenuation_db=60.0):
        """
        Append a quarter-band band-stop stage (non-decimating).

        Blocks [≈fs/8, ≈fs/4] of the current input rate using the polyphase
        structure H_BS(z) = H_LP(z)·H_LP(z²) + H_HP(z).
        Does NOT decimate — sample rate is unchanged.

        Args:
            order:          Allpass order (None = auto from attenuation_db).
            attenuation_db: Stopband attenuation [dB] when order is None.
        """
        from .design import quarterband_bs as _qbs_design
        fs   = self._current_input_fs()
        filt = _qbs_design(fs, order=order, attenuation_db=attenuation_db)
        spec = StageSpec(len(self.stages), 'BS', fs, fs / 8.0, filt.coefs, filt)
        self.stages.append(spec)
        return self

    # --- signal processing ---

    def apply(self, x):
        """Apply all stages sequentially (filter + decimate ×2 each)."""
        y = np.asarray(x, dtype=float)
        for stage in self.stages:
            y = stage.apply(y)
        return y

    # --- introspection ---

    def info(self):
        """Print a full summary of every stage."""
        ratio = 2 ** len(self.stages)
        print(f"\n{'#'*60}")
        print(f"  DecimatorChain   {self.fs:.1f} Hz → {self.output_fs:.1f} Hz")
        print(f"  {len(self.stages)} stage(s),  decimation ×{ratio}")
        print(f"{'#'*60}")
        for stage in self.stages:
            stage.info()
        print()

    def plot(self, n_points=8192, show_branches=True):
        """
        Plot per-stage responses and the overall cascade magnitude.

        Args:
            n_points:       FFT resolution.
            show_branches:  Overlay Branch X / Y on each stage plot.
        """
        n = len(self.stages)
        if n == 0:
            print("No stages to plot.")
            return

        n_rows = n + 1  # per-stage rows + cascade row
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.8 * n_rows),
                                 constrained_layout=True)
        fig.suptitle(
            f"DecimatorChain  {self.fs:.0f} Hz → {self.output_fs:.0f} Hz  "
            f"(×{2**n} decimation)",
            fontsize=13, fontweight='bold'
        )
        if n_rows == 1:
            axes = [axes]

        # --- per-stage sub-plots ---
        for i, stage in enumerate(self.stages):
            ax  = axes[i]
            fs  = stage.input_fs
            tf  = stage.get_transfer_function()

            freqs, h = _freqz(tf.b, tf.a, fs, n_points)
            ax.plot(freqs, _mag_db(h), color='steelblue', lw=2.0, label='Combined')

            if show_branches:
                bx_tf = stage._filter.bi.get_transfer_function()
                by_tf = stage._filter.by.get_transfer_function()
                for f_obj, lbl, col in [(bx_tf, 'Branch X', 'tomato'),
                                        (by_tf, 'Branch Y', 'seagreen')]:
                    fq, hb = _freqz(f_obj.b, f_obj.a, fs, n_points)
                    ax.plot(fq, _mag_db(hb), '--', color=col, lw=1.2, label=lbl)

            ax.axvline(stage.f_pass, color='orange', ls=':', lw=1.2,
                       label=f'f_pass={stage.f_pass:.1f} Hz')
            ax.set_title(
                f"Stage {i}: {stage.filter_type}   "
                f"{fs:.0f} Hz → {stage.output_fs:.0f} Hz,  "
                f"f_pass={stage.f_pass:.1f} Hz,  "
                f"order={len(stage.coefs)}"
            )
            ax.set_ylabel("Magnitude [dB]")
            ax.set_ylim(-80, 5)
            ax.set_xlim(0, fs / 2)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='lower left')

        # --- cascade plot ---
        ax_cas = axes[-1]
        ax_cas.set_title(
            f"Cascade magnitude  (referred to input fs={self.fs:.0f} Hz)")
        ax_cas.set_ylabel("Magnitude [dB]")
        ax_cas.set_xlabel("Frequency [Hz]")
        ax_cas.set_ylim(-80, 5)
        ax_cas.set_xlim(0, self.fs / 2)
        ax_cas.grid(True, alpha=0.3)

        # Build cascade on a common frequency grid [0, fs_in/2]
        common_freqs = np.linspace(0, self.fs / 2, n_points)
        cascade_db   = np.zeros(n_points)

        for stage in self.stages:
            # Evaluate this stage's response on its own [0, fs_stage/2] sub-grid
            valid = common_freqs <= stage.output_fs
            w_stage = 2 * np.pi * common_freqs[valid] / stage.input_fs
            tf = stage.get_transfer_function()
            _, h = freqz(tf.b, tf.a, worN=w_stage)
            cascade_db[valid] += _mag_db(h)

        ax_cas.plot(common_freqs, cascade_db, color='steelblue', lw=2.0)
        ax_cas.axvline(self.output_fs, color='orange', ls=':', lw=1.2,
                       label=f'output fs/2={self.output_fs:.1f} Hz')
        ax_cas.legend(fontsize=8)

        plt.show()
        return fig
