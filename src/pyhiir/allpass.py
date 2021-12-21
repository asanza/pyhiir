from numpy import polysub, zeros, polymul, polyadd
import numpy as np
from scipy.signal import lfilter

def FilterAdd(f1, f2):
    den = polymul(f1.get_den(), f2.get_den())
    n1 = polymul(f1.get_num(), f2.get_den())
    n2 = polymul(f1.get_den(), f2.get_num())
    num = polyadd(n1, n2)
    return Filter(num, den)

def FilterSub(f1, f2):
    den = polymul(f1.get_den(), f2.get_den())
    n1 = polymul(f1.get_num(), f2.get_den())
    n2 = polymul(f1.get_den(), f2.get_num())
    num = polysub(n1, n2)
    return Filter(num, den)

def FilterMult(f1, f2):
  return Filter(polymul(f1.get_num(), f2.get_num()), polymul(f1.get_den(), f2.get_den()))


class Filter:
  def __init__(self, b, a):
    self.b = b
    self.a = a

  def get_num(self):
    return self.b

  def get_den(self):
    return self.a

  def apply(self, x):
    return lfilter(self.b, self.a, x)

class AllPass:
  def __init__(self, order, b):
    self.b = zeros(order + 1)
    self.a = zeros(order + 1)

    self.b[0] = b
    self.b[order] = 1

    self.a[0] = 1
    self.a[order] = b

  def apply(self, x):
    return lfilter(self.b, self.a, x)

  def order(self):
    return len(self.a) - 1

  def get_transfer_function(self):
    return Filter(self.b, self.a)

class AllPassChain:
  def __init__(self, filter_list = []):
    self.filters = filter_list

  def order(self):
    return self.filters[0].order()

  def append(self, filter):
    self.filters.append(filter)

  def apply_chain(self, x):
    for i in self.filters:
      x = i.apply(x)

    return x

  def __str__(self):
      for i in self.filters:
          return "b: " + str(i.b) + " a: " + str(i.a)

  def get_transfer_function(self):
    den = 1
    num = 1
    for i in self.filters:
      den = polymul(den, i.a)
      num = polymul(num, i.b)
    return Filter(num, den)

class Delay:
  def __init__(self, order):
    self.order = order

  def get_transfer_function(self):
    den = zeros(np.int(self.order / 2) + 1)
    den[0] = 1
    return Filter(1, den)

  def get_den(self):
    return self.get_transfer_function().get_den()

  def get_num(self):
    return self.get_transfer_function().get_num()

class LowPass:
  def __init__(self, coef):
    bi = []
    by = []
    for i in range(0, len(coef), 2):
      bi.append(AllPass(2, coef[i]))
    for i in range(1, len(coef), 2):
      by.append(AllPass(2, coef[i]))
    self.bi = AllPassChain(bi)
    self.by = AllPassChain(by)

  def get_transfer_function(self):
    bi_tf = self.bi.get_transfer_function()
    by_tf = self.by.get_transfer_function()
    dl = Delay(self.bi.order()).get_transfer_function()

    by_tf = FilterMult(dl, by_tf)

    out = FilterAdd(bi_tf, by_tf)
    out = Filter(out.get_num(), 2 * out.get_den())

    return out

  def get_num(self):
    return self.get_transfer_function().get_num()

  def get_den(self):
    return self.get_transfer_function().get_den()

class HighPass:
  def __init__(self, coef):
    bi = []
    by = []
    for i in range(0, len(coef), 2):
      bi.append(AllPass(2, coef[i]))
    for i in range(1, len(coef), 2):
      by.append(AllPass(2, coef[i]))
    self.bi = AllPassChain(bi)
    self.by = AllPassChain(by)

  def get_transfer_function(self):
    bi_tf = self.bi.get_transfer_function()
    by_tf = self.by.get_transfer_function()
    dl = Delay(self.bi.order()).get_transfer_function()

    by_tf = FilterMult(dl, by_tf)

    out = FilterSub(bi_tf, by_tf)
    out = Filter(out.get_num(), 2 * out.get_den())

    return out

  def get_num(self):
    return self.get_transfer_function().get_num()

  def get_den(self):
    return self.get_transfer_function().get_den()
