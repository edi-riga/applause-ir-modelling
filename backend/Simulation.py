#!/usr/bin/env python3
import os
import numpy as np
from Model import Model

class Simulation():
  """
  A simulation consists of several models, each representing different parts
  of the system.
  
  Constructor accepts a list of model instances to be used in the simulation.
  The first model in the list should generate the initial simulation data,
  which then gets passed to consecutive models. Each model can be supplied
  with an optional argument value or a list of values. Since each of the
  argument values is likely to produce a different output, this splits the
  simulation into separate branches. The number of results at the end of the
  simulation is thus equal to the product of argument numbers of each model.

              arg1         [arg21, arg22]      [arg31, arg32]
               |                 |                   |
               |                 |                   v
               |                 |             --------------
               v                 v          -> |   Model 3  | -->
        --------------     --------------  /   | (Params 3) | -->
        |   Model 1  | --> |   Model 2  | -    --------------
        | (Params 1) |     | (Params 2) | -          v             . . .
        --------------     --------------  \   --------------
                                            -> |   Model 3  | -->
                                               | (Params 3) | -->
                                               --------------
  """
  def __init__(self, models, use_cache=True, cachedir='cache', displaydir='display'):
    self.models     = models
    self.use_cache  = use_cache
    self.cachedir   = cachedir
    self.displaydir = displaydir

    for m in self.models:
      assert issubclass(m.__class__, Model),\
        "Simulation components must inherit from 'Model' superclass"

    assert self.models[0].input_tuple == None,\
      "First model's input tuple should be None"

    for i in range(len(self.models)-1):
      assert self.is_compatible(self.models[i], self.models[i+1]) == True,\
        "Models '" + type(self.models[i]).__name__ + \
        "' : '"   + type(self.models[i+1]).__name__ + \
        "' are not compatible!"

    if use_cache and not os.path.exists(cachedir):
      os.makedirs(cachedir)

    if not os.path.exists(displaydir):
      for i in range(len(self.models)):
        if self.models[i].visualize:
          os.makedirs(displaydir)
          break


  def is_compatible(self, input_model, output_model):
    for key in output_model.input_tuple:

      if key not in input_model.output_tuple:
        print("ERROR: Type mismatch for '"
          + type(input_model).__name__ + "' and '"
          + type(output_model).__name__ + "' key '"
          + key + "' is not found")
        return False

      if input_model.output_tuple[key] != output_model.input_tuple[key]:
        print("ERROR: Shape mismatch for '"
          + key + "' key in '"
          + type(input_model).__name__ + "' and '"
          + type(output_model).__name__ + "'")
        return False

    return True

  def process(self, index=0, input_data=None, param_id=""):
    output_data = []

    if index >= len(self.models):
      return input_data

    for args in self.models[index].args_list:
      print("Executing: " + type(self.models[index]).__name__)

      param_str = self.models[index].get_parameter_id_str(args, input_data)
      param_id_new = param_id
      if param_str != None:
        param_id_new += param_str + '_'

      if self.use_cache:
        prefix = self.cachedir + '/' + str(index) + '_' + param_id_new + type(self.models[index]).__name__ + '_'
        intermediate_data = self.models[index].load(prefix, args, input_data)
        if isinstance(intermediate_data, type(None)):
          intermediate_data = self.models[index].process(input_data, args)
          self.models[index].store(prefix, args, input_data, intermediate_data)
      else:
        intermediate_data = self.models[index].process(input_data, args)

      if self.models[index].visualize:
        prefix = self.displaydir + '/' + str(index) + '_' + param_id_new + 'disp_' + type(self.models[index]).__name__ + '_'
        self.models[index].store_display(prefix, args, input_data, intermediate_data, self.use_cache)

      result = self.process(index+1, intermediate_data, param_id_new)
      output_data.append(result)

    return output_data
