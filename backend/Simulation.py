#!/usr/bin/env python3
import os
import numpy as np
from Model import Model

class Simulation():
  ''' TODO '''

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
