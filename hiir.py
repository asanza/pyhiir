#!/usr/bin/env python3

import ctypes as ct
import numpy as np
import os
from allpass import AllPass, AllPassChain, LowPass
from scipy import signal
import matplotlib.pyplot as plt

path = 'build/hiir/libhiir.so'

class hiir:
    def __init__(self):
        libname = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
        libname += path.replace('/', os.path.sep)

        self.lib = ct.cdll.LoadLibrary(libname)

        self.lib.iir2designer_compute_nbr_coefs_from_proto.restype = ct.c_int
        self.lib.iir2designer_compute_nbr_coefs_from_proto.argtypes = [ct.c_double, ct.c_double]

        self.lib.iir2designer_compute_coefs.restype = ct.c_int
        self.lib.iir2designer_compute_coefs.argtypes = [ct.POINTER(ct.c_double), ct.c_double, ct.c_double]

        self.lib.iir2designer_compute_coefs_spec_order_tbw.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.c_double]


    def compute_nbr_coefs_from_proto(self, attenuation, transition):
        return self.lib.iir2designer_compute_nbr_coefs_from_proto(ct.c_double(attenuation), ct.c_double(transition))

    def compute_coefs(self, attenuation, transition):
        n = self.compute_nbr_coefs_from_proto(attenuation, transition)
        x = np.zeros(n, dtype=np.double)
        self.lib.iir2designer_compute_coefs(x.ctypes.data_as(ct.POINTER(ct.c_double)).contents, ct.c_double(attenuation), ct.c_double(transition))
        return x

    def compute_coefs_order_tbw(self, n, transition):
        x = np.zeros(n, dtype=np.double)
        self.lib.iir2designer_compute_coefs_spec_order_tbw(x.ctypes.data_as(ct.POINTER(ct.c_double)).contents, ct.c_int(n), ct.c_double(transition))
        return x

if __name__=='__main__':
    h = hiir()
    #print(h.compute_nbr_coefs_from_proto(40, 0.1))
    x = h.compute_coefs_order_tbw(10, 0.1)
    f = LowPass(x)
    # print(x)    
    # #print(h.compute_coefs(40, 0.1))
    # ai = AllPass(2, x[0])
    # ay = AllPass(2, x[1])
    # ai1 = AllPass(2, x[2])
    # ay1 = AllPass(2, x[3])
    # ai2 = AllPass(2, x[4])
    # ay2 = AllPass(2, x[5])
    # ai3 = AllPass(2, x[6])
    # ay3 = AllPass(2, x[7])
    # ai4 = AllPass(2, x[8])
    # ay4 = AllPass(2, x[9])

    # bi = AllPassChain([ai, ai1, ai2, ai3, ai4])
    # by = AllPassChain([ay, ay1, ay2, ay3, ay4])

    # f = LowPass(bi, by)

    tf = f.get_transfer_function()

    w, h = signal.freqz(tf.b, tf.a)

    plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
    plt.show()