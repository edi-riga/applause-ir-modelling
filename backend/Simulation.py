#!/usr/bin/env python3
import numpy as np
from Model import Model

class Simulation():
  ''' TODO '''

  def __init__(self, models):
    self.models = models

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

  def is_compatible(self, input_model, output_model):
    if input_model.output_shape != output_model.input_shape:
      print("ERROR: Shape mismatch for '"
        + type(input_model).__name__ + "' and '"
        + type(output_model).__name__ + "'")
      return False

    return True
