#!/usr/bin/env python3
"""
Quick smoke-test for the new pyhiir toolchain API.
Run from the repo root:  python smoke_test.py
"""

from src.pyhiir import halfband, LowPass, HighPass, DecimatorChain

# ── 1. design.halfband: frequency specs → coefficients ─────────────────────
fs = 1500.0
coefs = halfband(fs=fs, f_pass=300, order=4)
print(f"halfband(fs={fs}, f_pass=300, order=4) → coefs = {coefs}")

# ── 2. LowPass: info + filter object ───────────────────────────────────────
lp = LowPass(coefs)
lp.info(fs=fs)

# ── 3. HighPass: info ──────────────────────────────────────────────────────
hp = HighPass(coefs)
hp.info(fs=fs)

# ── 4. DecimatorChain: 1500→375 Hz (two LP stages) ─────────────────────────
chain = (DecimatorChain(fs=1500)
         .add_lp(f_pass=300, order=4)    # 1500 → 750 Hz
         .add_lp(f_pass=150, order=2))   # 750  → 375 Hz
chain.info()

# ── 5. Apply to a 50 Hz test signal ────────────────────────────────────────
import numpy as np
t = np.arange(0, 0.5, 1/1500)
x = np.sin(2*np.pi*50*t)
y = chain.apply(x)
print(f"\nInput  length: {len(x)},  fs={chain.fs:.0f} Hz")
print(f"Output length: {len(y)},  fs={chain.output_fs:.0f} Hz")

# ── 6. plot (comment out if running headless) ───────────────────────────────
# lp.plot(fs=fs)
# chain.plot()
