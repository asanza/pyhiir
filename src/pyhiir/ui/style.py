"""Shared colours, Qt Fusion dark palette and matplotlib RC for pyhiir UI."""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

# ── Plot colours (used in matplotlib) ─────────────────────────────────────
C_COMBINED = "#4fc3f7"   # light blue
C_BRANCH_X = "#ef9a9a"   # soft red
C_BRANCH_Y = "#a5d6a7"   # soft green
C_PHASE    = "#ffcc80"   # amber
C_GD       = "#ce93d8"   # purple

PANEL      = "#2d2d2d"   # axes panel bg
BORDER     = "#555555"
TEXT       = "#e0e0e0"
MUTED      = "#888888"

# alias colours needed in app.py plot calls
ACCENT = C_COMBINED
GREEN  = C_BRANCH_Y
RED    = C_BRANCH_X
ORANGE = C_PHASE
PURPLE = C_GD

# Minimal stylesheet — only override the monospace code widget
STYLE = """
QTextEdit {
    font-family: 'JetBrains Mono','Fira Code','Consolas',monospace;
    font-size: 12px;
}
"""


def apply_dark_palette(app: QApplication) -> None:
    """Apply Qt Fusion dark palette — the standard Qt dark theme."""
    app.setStyle("Fusion")
    p = QPalette()
    dark   = QColor(45, 45, 45)
    darker = QColor(30, 30, 30)
    mid    = QColor(60, 60, 60)
    light  = QColor(80, 80, 80)
    text   = QColor(224, 224, 224)
    hi     = QColor(42, 130, 218)
    hi_txt = QColor(255, 255, 255)

    p.setColor(QPalette.Window,          dark)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            darker)
    p.setColor(QPalette.AlternateBase,   dark)
    p.setColor(QPalette.ToolTipBase,     darker)
    p.setColor(QPalette.ToolTipText,     text)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.Button,          mid)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Link,            hi)
    p.setColor(QPalette.Highlight,       hi)
    p.setColor(QPalette.HighlightedText, hi_txt)
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(128, 128, 128))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    app.setPalette(p)


# ── Matplotlib RC (dark axes matching Fusion dark) ─────────────────────────
MPL_RC = {
    "figure.facecolor": PANEL,
    "axes.facecolor":   "#1e1e1e",
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "axes.grid":        True,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "legend.facecolor": PANEL,
    "legend.edgecolor": BORDER,
    "legend.fontsize":  10,
    "lines.linewidth":  1.8,
}
