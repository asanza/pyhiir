# pyhiir
use hiir (https://github.com/unevens/hiir) library in python 

HIIR is a DSP (digital signal processing) library in C++ which allows:
 - Changing the sampling rate of a signal by a factor of 2
 - Obtaining two signals with a pi/2 phase difference (Hilbert Transform)

 Pyhiir offers wrappers for the hiir methods in python. 

## Usage Examples

### Hilbert Transformer
```python
from pyhiir.hiir import hiir
from pyhiir.allpass import Hilbert
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __name__=='__main__':
    h = hiir()

    # compute filter coeficients for 40db attenuation 
    # and transition band of 10% of the sampling frequency
    c = h.compute_coefs(40, .1)

    # create a hilbert transformer with allpass chains
    h = Hilbert(c)

    fi, fq = h.get_transfer_function()
    wi, hi = signal.freqz(fi.b, fi.a, fs=2)
    wq, hq = signal.freqz(fq.b, fq.a, fs=2)

    # Plot the frequency response
    plt.plot(wi, (np.angle(hi)), color='r')
    plt.plot(wq, (np.angle(hq)), color='g')
    
    # plt.plot(wi, (np.angle(hi) - np.angle(hq)), color='y')

    plt.ylabel('Phase [rad]', color='r')
    plt.xlabel('Frequency')
    plt.tick_params('y', colors='r')
    plt.show()
    
    plt.plot(wi, np.abs(hi), color='r')
    plt.plot(wq, np.abs(hq), color='g')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude [db]')
    plt.ylim(0.95, 1.05)
    plt.show()
```
![Filter Response](doc/hilphas.png)
![Filter Response](doc/hilamp.png)


### Decimator Filter:

```python
from pyhiir.hiir import hiir
from pyhiir.allpass import LowPass
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __name__=='__main__':
    h = hiir()

    # compute filter coeficients for 120db attenuation 
    # and transition band of 1% of the sampling frequency
    c = h.compute_coefs(120, .01)

    # create an half-band low pass filter with allpass chains
    f = LowPass(c)

    # Now get the filter transfer function for plotting
    ff = f.get_transfer_function()
    w, h = signal.freqz(ff.b, ff.a)
    plt.grid(True)
    plt.xlabel("Normalized frequency")
    plt.ylabel("Amplitude [dB]")
    plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
    plt.show()
```
![Filter Response](doc/filter1.png)

