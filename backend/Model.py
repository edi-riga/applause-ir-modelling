#!/usr/bin/env python3
import numpy as np

class Model():
  def __init__(self, input_shape=None, output_shape=None, args_list=[None]):
    self.input_shape  = input_shape
    self.output_shape = output_shape
    self.args_list    = args_list

  def set_args_list(self, args_list):
    self.args_list = args_list

  # To be implemented by the dervived class
  def process(self, input_data=None, args=None):
    raise NotImplementedError()

  # To be implemented by the derived class
  def store(self, prefix, args, data):
    raise NotImplementedError()

  # To be implemented by the derived class
  def load(self, prefix, args):
    raise NotImplementedError()
