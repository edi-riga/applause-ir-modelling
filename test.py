#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append('backend')
from Model import Model
from Simulation import Simulation

class Test0(Model):
  def __init__(self, input_shape=None, output_shape=(1,)):
    super().__init__(input_shape, output_shape)

class Test1(Model):
  def __init__(self, input_shape=(1,), output_shape=None):
    super().__init__(input_shape, output_shape)


test0 = Test0()
test1 = Test1()
tests = [test0, test1]
sim = Simulation(tests)
