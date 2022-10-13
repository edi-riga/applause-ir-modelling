#!/usr/bin/env python3
import numpy as np

class Model():
  def __init__(self, input_tuple=None, output_tuple=None, args_list=[None], visualize=False):
    self.input_tuple  = input_tuple
    self.output_tuple = output_tuple
    self.args_list    = args_list
    self.visualize    = visualize

  def set_args_list(self, args_list):
    self.args_list = args_list

  # To be implemented by the dervived class
  def process(self, input_data=None, args=None):
    raise NotImplementedError()

  # To be implemented by the derived class
  def store(self, prefix, args, input_data, output_data):
    raise NotImplementedError()

  # To be implemented by the derived class
  def load(self, prefix, args, input_data):
    raise NotImplementedError()

  # To be implemented by the derived class
  def store_display(self, prefix, args, input_data, output_data, cached):
    raise NotImplementedError()

  # To be implemented by the derived class
  def get_parameter_id_str(self, args, input_data):
    raise NotImplementedError()
