#!/usr/bin/env python3

from pyhiir.hiir import hiir
from pyhiir.allpass import LowPass
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

h = hiir()
x = h.compute_coefs_order_tbw(2, 0.1)
# f = LowPass(x)

exit(0)

# tf = f.get_transfer_function()

# w, h = signal.freqz(tf.b, tf.a)


# plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
# plt.show()