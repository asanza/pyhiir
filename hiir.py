import ctypes as ct
import numpy as np
import os

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

    def compute_nbr_coefs_from_proto(self, attenuation, transition):
        return self.lib.iir2designer_compute_nbr_coefs_from_proto(ct.c_double(attenuation), ct.c_double(transition))

    def compute_coefs(self, attenuation, transition):
        n = self.compute_nbr_coefs_from_proto(attenuation, transition)
        x = np.zeros(n, dtype=np.double)
        self.lib.iir2designer_compute_coefs(x.ctypes.data_as(ct.POINTER(ct.c_double)).contents, ct.c_double(attenuation), ct.c_double(transition))
        return x

if __name__=='__main__':
    h = hiir()
    #print(h.compute_nbr_coefs_from_proto(40, 0.1))
    print(h.compute_coefs(80, 0.001))