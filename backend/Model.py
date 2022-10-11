#!/usr/bin/env python3
import numpy as np

class Model():
  def __init__(self, input_shape=None, output_shape=None):
    self.input_shape  = input_shape
    self.output_shape = output_shape

  # To be implemented by the dervived class
  def process(self, input=None, output=None):
    raise NotImplementedError()

  # To be implemented by the derived class
  def save(self, filename):
    raise NotImplementedError()
