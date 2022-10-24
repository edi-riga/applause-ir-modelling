#!/usr/bin/env python3
import numpy as np

class Model():
  """
  Base class for simulation models.

  A model produces simulation data either by generating it or by
  transforming input data. Each model may have a set of parameters
  that are set during its instantiation and remain constant throughout
  the simulation. In addition, a model may accept a list of arguments
  that can be specified on each simulation run.
  
                         Argument(s)
                              |
                              v
                      ----------------
       Input data --> |     Model    | --> Output data
                      | (Parameters) |
                      ----------------
  """
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

  def store(self, prefix, args, input_data, output_data):
    for key in output_data:
      if self.output_tuple[key] == (1,):
        with open(f'{prefix}_{str(args)}_{key}', 'w') as f:
          f.write(str(output_data[key]))
      else:
        np.savetxt(f'{prefix}_{str(args)}_{key}', output_data[key])
  
  def load(self, prefix, args, input_data):
    data = {}
    try:
      for key, value in self.output_tuple.items():
        if value == (1,):
          with open(f'{prefix}_{str(args)}_{key}', 'r') as f:
            data[key] = float(f.read())
        else:
          data[key] = np.loadtxt(f'{prefix}_{str(args)}_{key}')
      return data
    except:
      return None

  # To be implemented by the derived class
  def store_display(self, prefix, args, input_data, output_data, cached):
    raise NotImplementedError()

  # To be implemented by the derived class
  def get_parameter_id_str(self, args, input_data):
    raise NotImplementedError()
