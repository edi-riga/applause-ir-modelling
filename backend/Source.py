#!/usr/bin/env python3
import numpy as np
from Model import Model


class Source(Model):
  def __init__(self, input_shape, output_shape):
    super().__init__(input_shape, output_shape)
