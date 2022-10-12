#!/usr/bin/env python3
import os
import numpy as np
from Model import Model

class Simulation():
  ''' TODO '''

  def __init__(self, models, use_cache=True, cachedir='cache'):
    self.models    = models
    self.use_cache = use_cache
    self.cachedir  = cachedir

    for m in self.models:
      assert issubclass(m.__class__, Model),\
        "Simulation components must inherit from 'Model' superclass"

    assert self.models[0].input_shape == None,\
      "First model's input shape should not exist"

    for i in range(len(self.models)-1):
      assert self.is_compatible(self.models[i], self.models[i+1]) == True,\
        "Models '" + type(self.models[i]).__name__ + \
        "' : '"   + type(self.models[i+1]).__name__ + \
        "' are not compatible!"

    if use_cache and not os.path.exists(cachedir):
      os.makedirs(cachedir)


  def is_compatible(self, input_model, output_model):
    if input_model.output_shape != output_model.input_shape:
      print("ERROR: Shape mismatch for '"
        + type(input_model).__name__ + "' and '"
        + type(output_model).__name__ + "'")
      return False

    return True

  def process(self, index=0, input_data=None):
    output_data = []

    if index >= len(self.models):
      return input_data

    for args in self.models[index].args_list:
      if self.use_cache:
        prefix = self.cachedir + '/' + str(index) + '_' + type(self.models[index]).__name__ + '_'
        intermediate_data = self.models[index].load(prefix, args)
        if intermediate_data == None:
          intermediate_data = self.models[index].process(input_data, args)
          self.models[index].store(prefix, args, intermediate_data)
      else:
        intermediate_data = self.models[index].process(input_data, args)

      result = self.process(index+1, intermediate_data)
      output_data.append(result)

    return output_data
