#!/usr/bin/env python3
"""pyhiir — Polyphase IIR Filter Designer  (PySide6)

Run from repo root:
    python pyhiir_ui.py
    python -m src.pyhiir.ui.app
"""

import sys, os
import numpy as np
from scipy.signal import freqz, group_delay as sp_gd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QTabWidget, QTextEdit, QListWidget, QSizePolicy, QCheckBox,
    QStatusBar, QScrollArea,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# ── imports: relative (package) or absolute (script) ─────────────────────
try:
    from ..allpass import (LowPass, HighPass, Hilbert, FirstOrderLP, FirstOrderHP,
                           FirstOrderBS, ButterworthFilter, DCBlocker, QuarterBandBS,
                           AllPassFirst, FilterMult)
    from ..design  import (halfband, first_order, first_order_bs, butterworth,
                           dc_blocker, quarterband_bs)
    from ..chain   import DecimatorChain
    from .style    import (STYLE, MPL_RC, apply_dark_palette,
        PANEL, BORDER, ACCENT, GREEN, RED, ORANGE, PURPLE,
        TEXT, MUTED, C_COMBINED, C_BRANCH_X, C_BRANCH_Y, C_PHASE, C_GD)
    from ..allpass import ascii_tf as _ascii_tf
except ImportError:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.pyhiir.allpass import (LowPass, HighPass, Hilbert, FirstOrderLP, FirstOrderHP,
                                    FirstOrderBS, ButterworthFilter, DCBlocker, QuarterBandBS,
                                    AllPassFirst, FilterMult)
    from src.pyhiir.design  import (halfband, first_order, first_order_bs, butterworth,
                                    dc_blocker, quarterband_bs)
    from src.pyhiir.chain   import DecimatorChain
    from src.pyhiir.ui.style import (STYLE, MPL_RC, apply_dark_palette,
        PANEL, BORDER, ACCENT, GREEN, RED, ORANGE, PURPLE,
        TEXT, MUTED, C_COMBINED, C_BRANCH_X, C_BRANCH_Y, C_PHASE, C_GD)
    from src.pyhiir.allpass import ascii_tf as _ascii_tf

import matplotlib
matplotlib.rcParams.update(MPL_RC)


# ═══════════════════════════════════════════════════════════════════════════
# Canvas
# ═══════════════════════════════════════════════════════════════════════════

class Canvas(FigureCanvasQTAgg):
    def __init__(self, nrows=1, figsize=(8, 3.8)):
        fig = Figure(figsize=figsize, facecolor=PANEL)
        super().__init__(fig)
        self.fig = fig
        self.axs = fig.subplots(nrows, 1, squeeze=False)[:, 0]
        for ax in self.axs:
            ax.set_facecolor("#1e1e1e")
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.tick_params(colors=MUTED)
        fig.tight_layout(pad=1.8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        for ax in self.axs:
            ax.cla()
            ax.set_facecolor("#1e1e1e")
            ax.grid(True, color=BORDER, linestyle="--", alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Left panel — Half-Band / Polyphase  (LP, HP, Hilbert, QuarterBandBS)
# ═══════════════════════════════════════════════════════════════════════════

class SingleFilterPanel(QWidget):
    designed = Signal(object, float)   # (filter_obj, fs)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        grp = QGroupBox("Filter Parameters")
        form = QFormLayout(grp)
        form.setSpacing(8)

        self.cb_type = QComboBox()
        self.cb_type.addItems([
            "LowPass  (half-band)",
            "HighPass (half-band)",
            "Hilbert  (90° splitter)",
        ])
        self.cb_type.currentIndexChanged.connect(self._type_changed)
        form.addRow("Type:", self.cb_type)

        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(1, 1e9); self.sp_fs.setValue(1500); self.sp_fs.setSuffix(" Hz")
        form.addRow("Fs:", self.sp_fs)

        # f_pass only for LP/HP (not needed for Hilbert or auto BS)
        self.sp_fpass = QDoubleSpinBox()
        self.sp_fpass.setRange(0.01, 1e8); self.sp_fpass.setValue(300)
        self.sp_fpass.setSuffix(" Hz")
        self.lbl_fpass = QLabel("f_pass:")
        form.addRow(self.lbl_fpass, self.sp_fpass)

        self.sp_fstop = QDoubleSpinBox()
        self.sp_fstop.setRange(0, 1e8); self.sp_fstop.setValue(0)
        self.sp_fstop.setSuffix(" Hz"); self.sp_fstop.setSpecialValueText("auto")
        self.lbl_fstop = QLabel("f_stop:")
        form.addRow(self.lbl_fstop, self.sp_fstop)

        # BS note
        self.bs_note = QLabel("Auto quarter-band: blocks [fs/8, fs/4].\nNot tunable — fixed by decimation tree.")
        self.bs_note.setStyleSheet("color: gray; font-size: 10px;")
        self.bs_note.setWordWrap(True)
        self.bs_note.setVisible(False)
        form.addRow(self.bs_note)

        self.cb_ordermode = QComboBox()
        self.cb_ordermode.addItems(["Fixed order", "Auto (attenuation)"])
        self.cb_ordermode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Order mode:", self.cb_ordermode)

        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 32); self.sp_order.setValue(4)
        form.addRow("Order:", self.sp_order)

        self.sp_atten = QDoubleSpinBox()
        self.sp_atten.setRange(10, 200); self.sp_atten.setValue(60)
        self.sp_atten.setSuffix(" dB"); self.sp_atten.setVisible(False)
        form.addRow("Attenuation:", self.sp_atten)

        lay.addWidget(grp)

        grp2 = QGroupBox("Plot options")
        olay = QVBoxLayout(grp2)
        self.chk_branches = QCheckBox("Show branches (X / Y)")
        self.chk_branches.setChecked(True)
        olay.addWidget(self.chk_branches)
        lay.addWidget(grp2)

        btn = QPushButton("⚡  Design Filter")
        btn.clicked.connect(self._design)
        lay.addWidget(btn)
        lay.addStretch()

    def _mode_changed(self, idx):
        self.sp_order.setVisible(idx == 0)
        self.sp_atten.setVisible(idx == 1)

    def _type_changed(self, idx):
        is_hilbert = (idx == 2)
        self.sp_fpass.setVisible(not is_hilbert)
        self.lbl_fpass.setVisible(not is_hilbert)
        self.sp_fstop.setVisible(not is_hilbert)
        self.lbl_fstop.setVisible(not is_hilbert)
        self.bs_note.setVisible(False)  # no BS in this tab anymore

    def _design(self):
        fs     = self.sp_fs.value()
        f_pass = self.sp_fpass.value()
        f_stop = self.sp_fstop.value() or None
        idx    = self.cb_type.currentIndex()
        try:
            if self.cb_ordermode.currentIndex() == 0:
                coefs = halfband(fs, f_pass, order=self.sp_order.value(), f_stop=f_stop)
            else:
                coefs = halfband(fs, f_pass, attenuation_db=self.sp_atten.value(), f_stop=f_stop)
            filt = [LowPass, HighPass, Hilbert][idx](coefs)
            self.designed.emit(filt, fs)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Design error", str(e))


# ═══════════════════════════════════════════════════════════════════════
# Left panel — 1st-Order Filters  (single-rate, no decimation)
# ═══════════════════════════════════════════════════════════════════════

_SCALAR_TYPES = [
    "LP  (1st-order allpass)",
    "HP  (1st-order allpass)",
    "BS  (band-stop: LP + HP)",
    "─" * 28,
    "LP  (Butterworth)",
    "HP  (Butterworth)",
    "─" * 28,
    "DC Blocker  H=(1−z⁻¹)/(1−r·z⁻¹)",
]
_SEP_IDXS = {3, 6}   # indices that are separators


class ScalarFiltersPanel(QWidget):
    """Single-rate filter design: LP/HP/BS (1st-order allpass), Butterworth, DC Blocker."""
    designed = Signal(object, float)   # (filt, fs)

    def __init__(self):
        super().__init__()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget()
        lay    = QVBoxLayout(inner)
        lay.setSpacing(10)
        scroll_lay = QVBoxLayout(self)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.addWidget(scroll)
        scroll.setWidget(inner)

        grp = QGroupBox("Filter Parameters")
        form = QFormLayout(grp)
        form.setSpacing(8)

        self.cb_type = QComboBox()
        self.cb_type.addItems(_SCALAR_TYPES)
        self.cb_type.currentIndexChanged.connect(self._type_changed)
        form.addRow("Type:", self.cb_type)

        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(1, 1e9); self.sp_fs.setValue(1500); self.sp_fs.setSuffix(" Hz")
        form.addRow("Fs:", self.sp_fs)

        # f_low / f_pass
        self.sp_fpass = QDoubleSpinBox()
        self.sp_fpass.setRange(0.01, 1e8); self.sp_fpass.setValue(300)
        self.sp_fpass.setSuffix(" Hz")
        self.lbl_fpass = QLabel("f_pass / f_low:")
        form.addRow(self.lbl_fpass, self.sp_fpass)

        # f_high (BS only)
        self.sp_fhigh = QDoubleSpinBox()
        self.sp_fhigh.setRange(0.01, 1e8); self.sp_fhigh.setValue(375)
        self.sp_fhigh.setSuffix(" Hz")
        self.sp_fhigh.setVisible(False)
        form.addRow("f_high (BS):", self.sp_fhigh)

        # Order / sections
        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 32); self.sp_order.setValue(1)
        self.lbl_order = QLabel("Order / sections:")
        form.addRow(self.lbl_order, self.sp_order)

        # DC Blocker r or f_c
        self.sp_fc = QDoubleSpinBox()
        self.sp_fc.setRange(0.001, 1000); self.sp_fc.setValue(10)
        self.sp_fc.setSuffix(" Hz"); self.sp_fc.setVisible(False)
        self.lbl_fc = QLabel("f_c (DC Blocker):")
        self.lbl_fc.setVisible(False)
        form.addRow(self.lbl_fc, self.sp_fc)

        lay.addWidget(grp)

        grp2 = QGroupBox("Plot options")
        olay = QVBoxLayout(grp2)
        self.chk_branches = QCheckBox("Show branches")
        self.chk_branches.setChecked(True)
        olay.addWidget(self.chk_branches)
        lay.addWidget(grp2)

        btn = QPushButton("⚡  Design Filter")
        btn.clicked.connect(self._design)
        lay.addWidget(btn)
        lay.addStretch()

    def _type_changed(self, idx):
        if idx in _SEP_IDXS:
            # snap to nearest valid item
            self.cb_type.setCurrentIndex(idx + 1)
            return
        is_dc  = (idx == 7)
        is_bs  = (idx == 2)
        is_bw  = (idx in (4, 5))
        is_ap  = (idx in (0, 1, 2))   # allpass-based
        self.sp_fhigh.setVisible(is_bs)
        self.sp_fc.setVisible(is_dc)
        self.lbl_fc.setVisible(is_dc)
        self.sp_fpass.setVisible(not is_dc)
        self.lbl_fpass.setVisible(not is_dc)
        self.sp_order.setVisible(not is_dc)
        self.lbl_order.setVisible(not is_dc)
        self.lbl_fpass.setText("f_low:" if is_bs else "f_pass:")
        if is_bw:
            self.lbl_order.setText("Order:")
        else:
            self.lbl_order.setText("Sections:")

    def _design(self):
        idx    = self.cb_type.currentIndex()
        fs     = self.sp_fs.value()
        f_pass = self.sp_fpass.value()
        n      = self.sp_order.value()
        try:
            if idx == 0:    # LP allpass
                filt = FirstOrderLP(first_order(fs, f_pass, n=n))
            elif idx == 1:  # HP allpass
                filt = FirstOrderHP(first_order(fs, f_pass, n=n))
            elif idx == 2:  # BS
                f_high = self.sp_fhigh.value()
                b_low, b_high = first_order_bs(fs, f_pass, f_high, n=n)
                filt = FirstOrderBS(b_low, b_high)
            elif idx == 4:  # Butterworth LP
                filt = butterworth(fs, f_pass, order=n, ftype='LP')
            elif idx == 5:  # Butterworth HP
                filt = butterworth(fs, f_pass, order=n, ftype='HP')
            elif idx == 7:  # DC Blocker
                filt = dc_blocker(fs, self.sp_fc.value())
            else:
                return
            self.designed.emit(filt, fs)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Design error", str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Left panel — Decimator Chain
# ═══════════════════════════════════════════════════════════════════════════

class ChainPanel(QWidget):
    designed = Signal(object)   # DecimatorChain

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        grp_fs = QGroupBox("Input sample rate")
        fl = QFormLayout(grp_fs)
        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(1, 1e9); self.sp_fs.setValue(1500); self.sp_fs.setSuffix(" Hz")
        fl.addRow("Fs:", self.sp_fs)
        lay.addWidget(grp_fs)

        grp_add = QGroupBox("Add stage")
        al = QFormLayout(grp_add)

        # f_pass explanation
        fpass_note = QLabel(
            "f_pass (polyphase): passband edge [Hz].\n"
            "  Must be \u003c fs/4. Stopband starts at fs/2\u2212f_pass.\n"
            "f_pass (1st-order): \u22123\u202fdB frequency [Hz]."
        )
        fpass_note.setStyleSheet("color: gray; font-size: 10px;")
        fpass_note.setWordWrap(True)
        al.addRow(fpass_note)

        self.sp_fpass = QDoubleSpinBox()
        self.sp_fpass.setRange(0.01, 1e8); self.sp_fpass.setValue(300); self.sp_fpass.setSuffix(" Hz")
        al.addRow("f_pass:", self.sp_fpass)

        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 32); self.sp_order.setValue(4)
        al.addRow("Order / sections:", self.sp_order)

        # ── Polyphase (half-band) row ──────────────────────────────────
        al.addRow(QLabel("Polyphase half-band  (2nd-order IIR, efficient):"))
        btn_row1 = QHBoxLayout()
        btn_lp = QPushButton("＋ LP")
        btn_hp = QPushButton("＋ HP")
        btn_lp.clicked.connect(lambda: self._add("LP"))
        btn_hp.clicked.connect(lambda: self._add("HP"))
        btn_row1.addWidget(btn_lp); btn_row1.addWidget(btn_hp)
        al.addRow(btn_row1)

        # ── BP / BS quarter-band row ───────────────────────────────────
        note = QLabel(
            "BP/BS = quarter-band (≈ fs/8 … fs/4).\n"
            "Not tunable — fixed by decimation tree."
        )
        note.setStyleSheet("color: gray; font-size: 11px;")
        note.setWordWrap(True)
        al.addRow(note)
        btn_bp = QPushButton("＋ BP  (quarter-band, auto)")
        btn_bp.clicked.connect(lambda: self._add("BP"))
        al.addRow(btn_bp)
        btn_bs = QPushButton("＋ BS  (quarter-band, no decim.)")
        btn_bs.clicked.connect(lambda: self._add("BS"))
        al.addRow(btn_bs)

        lay.addWidget(grp_add)

        grp_stages = QGroupBox("Stages")
        sl = QVBoxLayout(grp_stages)
        self.lst = QListWidget()
        self.lst.setMinimumHeight(120)
        sl.addWidget(self.lst)
        btn_rm = QPushButton("Remove selected"); btn_rm.setObjectName("danger")
        btn_rm.clicked.connect(self._remove)
        btn_cl = QPushButton("Clear all"); btn_cl.setObjectName("danger")
        btn_cl.clicked.connect(self._clear)
        sl.addWidget(btn_rm); sl.addWidget(btn_cl)
        lay.addWidget(grp_stages)

        btn_des = QPushButton("⚡  Build Chain")
        btn_des.clicked.connect(self._design)
        lay.addWidget(btn_des)
        lay.addStretch()

        self._stages = []   # list of (type, f_pass_low, f_pass_high, order)

    def _current_fs(self):
        fs = self.sp_fs.value()
        for s in self._stages:
            ft = s[0]
            if ft == "BP":
                fs /= 4
            elif ft != "BS":   # BS does not decimate
                fs /= 2
        return fs

    def _add(self, ftype):
        in_fs  = self._current_fs()
        order  = self.sp_order.value()
        if ftype == "BP":
            out_fs = in_fs / 4
            tbw   = 0.1
            f_lo  = round((in_fs / 2) * (0.25 - tbw / 2), 2)
            f_hi  = round(in_fs * (0.25 - tbw / 2), 2)
            self._stages.append((ftype, f_lo, f_hi, order))
            self.lst.addItem(
                f"Stage {len(self._stages)-1}: BP  "
                f"{in_fs:.1f}→{out_fs:.1f} Hz  "
                f"≈[{f_lo:.1f}–{f_hi:.1f} Hz]  order={order}"
            )
        elif ftype == "BS":
            self._stages.append((ftype, None, None, order))
            self.lst.addItem(
                f"Stage {len(self._stages)-1}: BS  "
                f"{in_fs:.1f} Hz (no decim.)  "
                f"stops[{in_fs/8:.1f}–{in_fs/4:.1f} Hz]  order={order}"
            )
        else:
            out_fs = in_fs / 2
            fp_lo  = self.sp_fpass.value()
            self._stages.append((ftype, fp_lo, None, order))
            self.lst.addItem(
                f"Stage {len(self._stages)-1}: {ftype}  "
                f"{in_fs:.1f}→{out_fs:.1f} Hz  f_pass={fp_lo:.1f}  order={order}"
            )

    def _remove(self):
        row = self.lst.currentRow()
        if row >= 0:
            self._stages.pop(row)
            self.lst.takeItem(row)
            self._refresh_labels()

    def _clear(self):
        self._stages.clear()
        self.lst.clear()

    def _refresh_labels(self):
        self.lst.clear()
        fs = self.sp_fs.value()
        dec = 1
        for i, entry in enumerate(self._stages):
            ft, fp_lo, fp_hi, order = entry
            in_fs = fs / dec
            if ft == "BP":
                out_fs = in_fs / 4; dec *= 4
                self.lst.addItem(
                    f"Stage {i}: BP  {in_fs:.1f}→{out_fs:.1f} Hz  "
                    f"[{fp_lo:.1f}–{fp_hi:.1f} Hz]  order={order}"
                )
            elif ft == "BS":
                self.lst.addItem(
                    f"Stage {i}: BS  {in_fs:.1f} Hz (no decim.)  "
                    f"stops[{in_fs/8:.1f}–{in_fs/4:.1f} Hz]  order={order}"
                )
                # dec unchanged
            else:
                out_fs = in_fs / 2; dec *= 2
                self.lst.addItem(
                    f"Stage {i}: {ft}  {in_fs:.1f}→{out_fs:.1f} Hz  "
                    f"f_pass={fp_lo:.1f}  order={order}"
                )

    def _design(self):
        if not self._stages:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No stages", "Add at least one stage.")
            return
        try:
            chain = DecimatorChain(fs=self.sp_fs.value())
            for ft, fp_lo, fp_hi, order in self._stages:
                if ft == "LP":
                    chain.add_lp(fp_lo, order=order)
                elif ft == "HP":
                    chain.add_hp(fp_lo, order=order)
                elif ft == "BP":
                    chain.add_quarterband_bp(order=order)
                elif ft == "BS":
                    chain.add_quarterband_bs(order=order)
            self.designed.emit(chain)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Design error", str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Right top — Plot panel
# ═══════════════════════════════════════════════════════════════════════════

class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # Magnitude tab
        self.cv_mag = Canvas(nrows=1)
        self.tb_mag = NavigationToolbar2QT(self.cv_mag, self)
        w1 = QWidget(); v1 = QVBoxLayout(w1); v1.addWidget(self.tb_mag); v1.addWidget(self.cv_mag)
        self.tabs.addTab(w1, "Magnitude")

        # Phase tab
        self.cv_ph = Canvas(nrows=1)
        self.tb_ph = NavigationToolbar2QT(self.cv_ph, self)
        w2 = QWidget(); v2 = QVBoxLayout(w2); v2.addWidget(self.tb_ph); v2.addWidget(self.cv_ph)
        self.tabs.addTab(w2, "Phase")

        # Group delay tab
        self.cv_gd = Canvas(nrows=1)
        self.tb_gd = NavigationToolbar2QT(self.cv_gd, self)
        w3 = QWidget(); v3 = QVBoxLayout(w3); v3.addWidget(self.tb_gd); v3.addWidget(self.cv_gd)
        self.tabs.addTab(w3, "Group Delay")

    def _freqz(self, b, a, fs, n=8192):
        w, h = freqz(b, a, worN=n)
        return w * fs / (2 * np.pi), h

    def plot_single(self, filt, fs, show_branches=True):
        """Plot LowPass, HighPass, or Hilbert."""
        self.cv_mag.clear(); self.cv_ph.clear(); self.cv_gd.clear()
        ax_m = self.cv_mag.axs[0]
        ax_p = self.cv_ph.axs[0]
        ax_g = self.cv_gd.axs[0]

        is_hilbert     = isinstance(filt, Hilbert)
        is_first_order = isinstance(filt, (FirstOrderLP, FirstOrderHP))
        is_bs          = isinstance(filt, FirstOrderBS)
        is_qbs         = isinstance(filt, QuarterBandBS)

        if is_hilbert:
            fi, fq = filt.get_transfer_function()
            pairs = [(fi, "I branch", C_COMBINED), (fq, "Q branch", C_BRANCH_Y)]
        elif is_qbs:
            tf = filt.get_transfer_function()
            pairs = [(tf, "Combined (BS)", C_COMBINED)]
            if show_branches:
                tf_lp = filt._lp.get_transfer_function()
                tf_hp = filt._hp.get_transfer_function()
                pairs += [(tf_lp, "LP narrow (≈fs/8)", C_BRANCH_X),
                          (tf_hp, "HP wide (≈fs/4)", C_BRANCH_Y)]
        elif is_bs:
            tf = filt.get_transfer_function()
            pairs = [(tf, "Combined (BS)", C_COMBINED)]
            if show_branches:
                tf_lp = filt._lp.get_transfer_function()
                tf_hp = filt._hp.get_transfer_function()
                pairs += [(tf_lp, "LP branch", C_BRANCH_X),
                          (tf_hp, "HP branch", C_BRANCH_Y)]
        elif is_first_order:
            tf = filt.get_transfer_function()
            pairs = [(tf, "Combined", C_COMBINED)]
            if show_branches:
                sections = [AllPassFirst(b) for b in filt.coefs]
                a_tf = sections[0].get_transfer_function()
                for s in sections[1:]:
                    a_tf = FilterMult(a_tf, s.get_transfer_function())
                pairs.append((a_tf, "Allpass A(z)", C_BRANCH_X))
        else:
            tf  = filt.get_transfer_function()
            pairs = [(tf, "Combined", C_COMBINED)]
            if show_branches:
                pairs += [
                    (filt.bi.get_transfer_function(), "Branch X", C_BRANCH_X),
                    (filt.by.get_transfer_function(), "Branch Y", C_BRANCH_Y),
                ]

        for f_obj, lbl, col in pairs:
            freqs, h = self._freqz(f_obj.b, f_obj.a, fs)
            mag_db   = 20 * np.log10(np.abs(h) + 1e-12)
            lw = 2.0 if "Combined" in lbl or "branch" in lbl else 1.2
            ls = "-" if "Combined" in lbl or "branch" in lbl else "--"
            ax_m.plot(freqs, mag_db, color=col, lw=lw, ls=ls, label=lbl)
            ax_p.plot(freqs, np.degrees(np.unwrap(np.angle(h))), color=col, lw=lw, ls=ls, label=lbl)
            try:
                _, gd = sp_gd(f_obj.b, f_obj.a, worN=8192)
                gd_f  = np.linspace(0, fs / 2, 8192)
                ax_g.plot(gd_f, gd, color=col, lw=lw, ls=ls, label=lbl)
            except Exception:
                pass

        title = f"{type(filt).__name__}   fs={fs:.0f} Hz"
        for ax, ylabel in [(ax_m, "Magnitude [dB]"), (ax_p, "Phase [°]"), (ax_g, "Group delay [samples]")]:
            ax.set_title(title, pad=6)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel(ylabel)
            if ax.get_legend_handles_labels()[1]:
                ax.legend()
            ax.set_xlim(0, fs / 2)
        ax_m.set_ylim(-80, 5)

        for cv in (self.cv_mag, self.cv_ph, self.cv_gd):
            cv.fig.tight_layout(pad=1.8)
            cv.draw()

    def plot_chain(self, chain):
        """Plot cascade response + per-stage overlay."""
        self.cv_mag.clear(); self.cv_ph.clear(); self.cv_gd.clear()
        ax_m = self.cv_mag.axs[0]

        colors = [ACCENT, GREEN, ORANGE, PURPLE, RED]
        common = np.linspace(0, chain.fs / 2, 8192)
        cascade_db = np.zeros(len(common))

        for i, stage in enumerate(chain.stages):
            col  = colors[i % len(colors)]
            tf   = stage.get_transfer_function()
            valid = common <= stage.output_fs
            w_s  = 2 * np.pi * common[valid] / stage.input_fs
            _, h = freqz(tf.b, tf.a, worN=w_s)
            db   = 20 * np.log10(np.abs(h) + 1e-12)
            cascade_db[valid] += db
            ax_m.plot(
                common[valid], db, "--", color=col, lw=1.2, alpha=0.7,
                label=f"Stage {i} {stage.filter_type} {stage.input_fs:.0f}→{stage.output_fs:.0f} Hz"
            )

        ax_m.plot(common, cascade_db, color=C_COMBINED, lw=2.2, label="Cascade")
        ax_m.axvline(chain.output_fs, color=ORANGE, ls=":", lw=1.2,
                     label=f"out Nyquist {chain.output_fs:.1f} Hz")
        ax_m.set_title(
            f"DecimatorChain  {chain.fs:.0f}→{chain.output_fs:.0f} Hz  "
            f"×{2**len(chain.stages)} decimation",
            color=TEXT, pad=6
        )
        ax_m.set_xlabel("Frequency [Hz]", color=TEXT)
        ax_m.set_ylabel("Magnitude [dB]", color=TEXT)
        ax_m.set_ylim(-80, 5)
        ax_m.set_xlim(0, chain.fs / 2)
        if ax_m.get_legend_handles_labels()[1]:
            ax_m.legend(fontsize=9)

        self.cv_mag.fig.tight_layout(pad=1.8)
        self.cv_mag.draw()


# ═══════════════════════════════════════════════════════════════════════════
# Right bottom — Coefficient display
# ═══════════════════════════════════════════════════════════════════════════

class CoefPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("Coefficients")
        lbl.setStyleSheet(f"color:{ACCENT}; font-weight:700; font-size:13px;")
        lay.addWidget(lbl)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        lay.addWidget(self.txt)

    def _arr(self, a):
        return np.array2string(np.asarray(a, dtype=float), precision=10,
                               separator=", ", max_line_width=72)

    def show_single(self, filt, fs):
        is_qbs         = isinstance(filt, QuarterBandBS)
        is_bs1         = isinstance(filt, FirstOrderBS)
        is_first_order = isinstance(filt, (FirstOrderLP, FirstOrderHP))
        is_hilbert     = isinstance(filt, Hilbert)

        if is_qbs:
            # ── QuarterBandBS  H_LP(z²) ──────────────────────────────
            lines  = [f"QuarterBandBS   H_LP(z\u00b2)   fs={fs:.1f} Hz", "",
                      f"Stopband  \u2248 [{fs/8:.1f}\u2013{3*fs/8:.1f}] Hz  (Q2 + Q3)",
                      f"Passband: [0\u2013{fs/8:.1f}] \u222a [{3*fs/8:.1f}\u2013{fs/2:.1f}] Hz  (Q1 + Q4)",
                      f"Null:       exactly {fs/4:.1f} Hz  (\u2212\u221e dB)",
                      f"-3 dB:      {fs/8:.1f} Hz  and  {3*fs/8:.1f} Hz  (halfband identity)",
                      "",
                      "H_BS(z) = H_LP(z\u00b2)   \u2190 z\u2192z\u00b2 halves the LP cutoff,",
                      "                       making the response symmetric around fs/4",
                      ""]
            lines += ["# Underlying LP stage (half-band at fs/4):"]
            lines += [_ascii_tf(filt._lp, fs=fs), ""]
            lines += ["# " + "\u2500" * 55]
            lines += [f"coefs = {self._arr(filt.coefs)}", ""]
            tf = filt.get_transfer_function()
            lines += ["# H_BS = H_LP(z\u00b2)  combined b/a:",
                      f"b = {self._arr(tf.b)}", f"a = {self._arr(tf.a)}"]
        elif is_bs1:
            # ── Band-stop ASCII TF ───────────────────────────────────
            lines  = [f"FirstOrderBS   fs={fs:.1f} Hz", ""]
            lines += ["H_BS(z) = H_LP(f_low) + H_HP(f_high)", ""]
            for label, coefs in [("LP branch  [A_low(z)]", filt.coefs_low),
                                  ("HP branch  [A_high(z)]", filt.coefs_high)]:
                lines += [f"{label}:"]
                for k, b in enumerate(coefs):
                    bs  = f"{b:.8f}"
                    lines += [f"  H{k}: ({bs} + z\u207b\u00b9) / (1 + {bs}\u00b7z\u207b\u00b9)"]
                lines.append("")
            lines += ["H_BS(z) = H_LP(z) + H_HP(z)    ← passes outside [f_low, f_high]", ""]
            lines += ["# " + "\u2500" * 55]
            lines += [f"b_low  = {self._arr(filt.coefs_low)}",
                      f"b_high = {self._arr(filt.coefs_high)}", ""]
            tf = filt.get_transfer_function()
            lines += ["# Combined", f"b = {self._arr(tf.b)}", f"a = {self._arr(tf.a)}"]
        elif is_first_order:
            # ── ASCII TF (first-order allpass form) ──────────────────
            ftype = type(filt).__name__
            op    = "+" if isinstance(filt, FirstOrderLP) else "\u2212"
            lines  = [f"{ftype}   sections={len(filt.coefs)}   fs={fs:.1f} Hz", ""]
            lines += ["Allpass branch:  A(z) = \u220f H\u1d62(z)", ""]
            for k, b in enumerate(filt.coefs):
                bs  = f"{b:.8f}"
                top = f"({bs} + z\u207b\u00b9)"
                bot = f"(1 + {bs}\u00b7z\u207b\u00b9)"
                lines += [f"  H{k}: {top}", f"       {bot}", ""]
            lines += [f"Combined:  H(z) = [ 1 {op} A(z) ] / 2", ""]
            lines += ["# " + "\u2500" * 55]
            lines += [f"branch_coefs = {self._arr(filt.coefs)}", ""]
            tf = filt.get_transfer_function()
            lines += ["# Combined", f"b = {self._arr(tf.b)}", f"a = {self._arr(tf.a)}"]
        elif isinstance(filt, (ButterworthFilter, DCBlocker)):
            # ── Generic b/a display (Butterworth or DC Blocker) ──────
            tf   = filt.get_transfer_function()
            name = type(filt).__name__
            if isinstance(filt, DCBlocker):
                fc_hz = filt.f_c * fs
                lines = [f"DCBlocker   r={filt.r:.8f}   fs={fs:.1f} Hz",
                         f"Approx. f_c \u2248 {fc_hz:.2f} Hz", "",
                         "H(z) = (1 \u2212 z\u207b\u00b9) / (1 \u2212 r\u00b7z\u207b\u00b9)", ""]
            else:
                lines = [f"Butterworth {filt.ftype}   order={filt._order}   "
                         f"f_pass={filt.f_pass:.2f} Hz   fs={fs:.1f} Hz", ""]
            lines += ["# " + "\u2500" * 55]
            if isinstance(filt, DCBlocker):
                lines += [f"r = {filt.r:.10f}", ""]
            lines += [f"b = {self._arr(tf.b)}", f"a = {self._arr(tf.a)}"]
        else:
            # ── ASCII TF (polyphase, 2nd-order sections) ─────────────
            lines = [_ascii_tf(filt, fs=fs), ""]
            lines += ["# " + "\u2500" * 55]
            lines += [f"branch_x_coefs = {self._arr(filt.branch_x_coefs)}"]
            lines += [f"branch_y_coefs = {self._arr(filt.branch_y_coefs)}", ""]
            if is_hilbert:
                fi, fq = filt.get_transfer_function()
                lines += ["# I-branch",
                          f"b_i = {self._arr(fi.b)}", f"a_i = {self._arr(fi.a)}", "",
                          "# Q-branch",
                          f"b_q = {self._arr(fq.b)}", f"a_q = {self._arr(fq.a)}"]
            else:
                tf = filt.get_transfer_function()
                lines += ["# Combined", f"b = {self._arr(tf.b)}", f"a = {self._arr(tf.a)}"]

        self.txt.setPlainText("\n".join(lines))

    def show_chain(self, chain):
        sep = "\u2500" * 58
        hdr = (f"DecimatorChain  {chain.fs:.1f} Hz \u2192 {chain.output_fs:.1f} Hz  "
               f"\u00d7{int(chain.fs / chain.output_fs)} decimation  "
               f"{len(chain.stages)} stage(s)")
        lines = [hdr, sep, ""]
        for s in chain.stages:
            tf = s.get_transfer_function()
            lines += [f"Stage {s.stage_idx}: {s.filter_type}  "
                      f"{s.input_fs:.1f} \u2192 {s.output_fs:.1f} Hz  "
                      f"f_pass={s.f_pass:.2f} Hz", ""]
            # ASCII TF
            if s._is_first_order:
                op = "+" if s.filter_type == "LP1" else "\u2212"
                for k, b in enumerate(s.coefs):
                    bs = f"{b:.8f}"
                    lines.append(
                        f"  H{k}: ({bs} + z\u207b\u00b9) / (1 + {bs}\u00b7z\u207b\u00b9)"
                    )
                lines += [f"  Combined: [ 1 {op} A(z) ] / 2", ""]
            else:
                lines.append(_ascii_tf(s._filter, fs=s.input_fs))
                lines.append("")
            # Raw coefficients
            if s._is_first_order:
                lines += [f"  b_{s.stage_idx} = {self._arr(s.coefs)}"]
            else:
                lines += [
                    f"  branch_x_{s.stage_idx} = {self._arr(s.branch_x_coefs)}",
                    f"  branch_y_{s.stage_idx} = {self._arr(s.branch_y_coefs)}",
                ]
            lines += [f"  b_{s.stage_idx} = {self._arr(tf.b)}",
                      f"  a_{s.stage_idx} = {self._arr(tf.a)}",
                      sep, ""]
        self.txt.setPlainText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyhiir — Polyphase IIR Filter Designer")
        self.resize(1280, 780)

        # ── status bar ────────────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — design a filter or build a chain.")

        # ── left tab widget (Single / Chain) ──────────────────────────────
        self.left_tabs = QTabWidget()

        self.single_panel = SingleFilterPanel()
        self.scalar_panel = ScalarFiltersPanel()
        self.chain_panel  = ChainPanel()
        self.left_tabs.addTab(self.single_panel, "Half-Band")
        # self.left_tabs.addTab(self.scalar_panel, "1st-Order")  # WIP — hidden
        self.left_tabs.addTab(self.chain_panel,  "Decimator Chain")

        self.single_panel.designed.connect(self._on_single)
        # self.scalar_panel.designed.connect(self._on_scalar)   # WIP — hidden
        self.chain_panel.designed.connect(self._on_chain)

        # ── right area ────────────────────────────────────────────────────
        self.plot_panel = PlotPanel()
        self.coef_panel = CoefPanel()
        self.coef_panel.setFixedHeight(220)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(self.plot_panel, stretch=1)
        rl.addWidget(self.coef_panel)

        # ── left panel: title label + tabs ────────────────────────────────
        left_container = QWidget()
        left_container.setFixedWidth(340)
        lc_lay = QVBoxLayout(left_container)
        lc_lay.setContentsMargins(0, 0, 0, 0)
        lc_lay.setSpacing(2)
        title_lbl = QLabel("pyhiir  ·  Filter Designer")
        title_lbl.setStyleSheet("font-size:13px; font-weight:700; padding:5px 8px;")
        title_lbl.setAlignment(Qt.AlignCenter)
        lc_lay.addWidget(title_lbl)
        lc_lay.addWidget(self.left_tabs)

        # ── splitter ──────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _add_header(self):
        pass  # title is now a label above the tab widget

    def _on_single(self, filt, fs):
        """Handler for Half-Band tab."""
        show_br = self.single_panel.chk_branches.isChecked()
        self.plot_panel.plot_single(filt, fs, show_branches=show_br)
        self.coef_panel.show_single(filt, fs)
        if isinstance(filt, QuarterBandBS):
            msg = (f"QuarterBandBS  H_LP(z\u00b2)  fs={fs:.0f} Hz  "
                   f"stopband\u2248[{fs/8:.1f}\u2013{3*fs/8:.1f}] Hz  "
                   f"order={len(filt.coefs)}")
        else:
            msg = (f"{type(filt).__name__}  fs={fs:.0f} Hz  "
                   f"order={len(filt.coefs)}  "
                   f"X={list(np.round(filt.branch_x_coefs, 6))}  "
                   f"Y={list(np.round(filt.branch_y_coefs, 6))}")
        self.status.showMessage(msg)

    def _on_scalar(self, filt, fs):
        """Handler for 1st-Order tab: any filter with get_transfer_function()."""
        show_br = self.scalar_panel.chk_branches.isChecked()
        self.plot_panel.plot_single(filt, fs, show_branches=show_br)
        self.coef_panel.show_single(filt, fs)
        name = type(filt).__name__
        if isinstance(filt, FirstOrderBS):
            msg = (f"{name}  fs={fs:.0f} Hz  "
                   f"b_low={list(np.round(filt.coefs_low,6))}  "
                   f"b_high={list(np.round(filt.coefs_high,6))}")
        elif isinstance(filt, DCBlocker):
            msg = f"DCBlocker  r={filt.r:.8f}  fs={fs:.0f} Hz"
        elif isinstance(filt, ButterworthFilter):
            msg = (f"{name} {filt.ftype}  order={filt._order}  "
                   f"f_pass={filt.f_pass:.1f} Hz  fs={fs:.0f} Hz")
        else:
            msg = (f"{name}  fs={fs:.0f} Hz  "
                   f"b={list(np.round(filt.coefs,6))}")
        self.status.showMessage(msg)

    def _on_chain(self, chain):
        self.plot_panel.plot_chain(chain)
        self.coef_panel.show_chain(chain)
        self.status.showMessage(
            f"Chain: {chain.fs:.0f}→{chain.output_fs:.0f} Hz  "
            f"×{2**len(chain.stages)} decimation  "
            f"{len(chain.stages)} stage(s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)       # Qt Fusion dark theme
    app.setStyleSheet(STYLE)      # minimal override (mono font for QTextEdit)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
