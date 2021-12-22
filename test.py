#!/usr/bin/env python3

from numpy.core.numeric import cross
from src.pyhiir.hiir import hiir
from src.pyhiir.allpass import Hilbert, LowPass
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

fg = 50 # grid freq
# max harmonic @ 60Hz
mh = 10

tf = .2
# sampling freq to capture the 10 harmonic
fs = 1500 # 1500 Hz
# max low freq
fmax = 60
# number of halbing stages
ndec = int(np.log2(fs/fmax))

fmax = fs / 2 ** ndec

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_crossings(sig, fs):
    """Estimate frequency by counting zero crossings
    """ 
    crossings = np.where(np.diff(sig))[0]
    crs = np.diff(crossings) / (0.5 * fs)
    t = np.arange(0, len(sig)/(fs), 1/fs)
    plt.plot(t, sig, '*')
    cros = np.diff(np.sign(sig))
    cros = cros.nonzero()[0]
    tt = np.arange(0, max(t), max(t)/len(cros))
    # cros = np.diff(cros)
    plt.plot(tt[0:], cros, '.')
    return 1 / crs    

def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * signal.blackmanharris(len(sig))
    f = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def freq_from_hilbert(sig, fs):
    s = signal.hilbert(sig)
    p = np.unwrap(np.angle(s))
    f = fs * np.diff(p) / (2.0 * np.pi)
    return f

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__=='__main__':

    print('number of decimator stages {0}'.format(ndec))
    print('max. measurable frequency  {0}'.format(fmax))
    
    t = np.arange(0, tf, 1/fs)
    x = np.sin(t * 2 * np.pi * fg) #+ np.random.normal(0, .5, len(t))

    # appling first decimator
    h = hiir()
    c = h.compute_coefs_order_tbw(4, .1)
    f = LowPass(c)
    ff = f.get_transfer_function()

    w, h = signal.freqz(ff.b, ff.a, fs=fs)
    plt.semilogy(w, np.abs(h))
    plt.show()

    y = signal.lfilter(ff.b, ff.a, x)
    fsf = fs
    y = y[1::2]
    fsf = fs / 2
    for i in range(0, 2):
        y = signal.lfilter(ff.b, ff.a, y)
        y = y[1::2]
        fsf = fsf / 2

    f = hiir()
    fg = 50.001
    fs = 500
    c = f.compute_coefs_order_tbw(4, .1)

    h = Hilbert(c)
    #k = LowPass(c).get_transfer_function()

    fi, fq = h.get_transfer_function()
    wi, hi = signal.freqz(fi.b, fi.a, 50000, fs=fs)
    wq, hq = signal.freqz(fq.b, fq.a, 50000, fs=fs)
    # Plot the frequency response
    plt.plot(wi, (np.angle(hi)), color='r')
    plt.plot(wq, (np.angle(hq)), color='g')
    #plt.plot(wi, (np.angle(hi) - np.angle(hq)), color='y')
    plt.ylabel('Phase [rad]', color='r')
    plt.tick_params('y', colors='r')
    plt.show()
    
    plt.plot(wi, np.abs(hi), color='r')
    plt.plot(wq, np.abs(hq), color='g')
    plt.show()


    plt.show()

