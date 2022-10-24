import os
import numpy as np
from PIL import Image

import sys
sys.path.append('../backend')
from Model import Model
import params


class NUC(Model):
  """
  Performs a linear non-uniformity correction.

  Input data
  ----------
  ADC :
      Image data.
  
  Output data
  -----------
  ADC :
      Corrected image.

  Initializer parameters
  ----------------------
  coef_a :
      Array of gain coefficients.
  
  coef_b :
      Array of offset coefficients.
  
  resolution :
      Resolution of the image.
  
  adc_resolution :
      Intensity value resolution (number of bits).
 """
  def __init__(self, coef_a, coef_b, resolution=params.resolution, adc_resolution=params.adc_resolution, visualize=False):
    self.size_h = resolution[0]
    self.size_v = resolution[1]
    self.coef_a = coef_a
    self.coef_b = coef_b
    self.adc_resolution = adc_resolution

    super().__init__(
      input_tuple = {
        "ADC" : (self.size_h, self.size_v)
      },
      output_tuple = {
        "ADC" : (self.size_h, self.size_v)
      },
      visualize = visualize
    )
  
  @staticmethod
  def calculate_coefs(frames, temps, fpart_width=params.nuc_fpart):
    """
    Calculates the coefficients for linear nonuniformity correction.
    
    Parameters
    ----------
    frames :
        List of frames observing a uniform target (black body radiator).
    
    temps :
        List of expected uniform intensity values per frame.
    
    fpart_width :
        Coefficient fractional part bit width.

    Returns
    -------
    coef_a :
        Array of gain coefficients.
    
    coef_b :
        Array of offset coefficients.
    """
    frame0 = frames[0]
    frame1 = frames[1]
    temp0  = temps[0]
    temp1  = temps[1]
    nuc_a = (temp0 - temp1) / (frame0 - frame1)
    nuc_b = (temp1*frame0 - temp0*frame1) / (frame0 - frame1)
    if fpart_width:
      step = 2**-fpart_width
      nuc_a = np.round(nuc_a / step) * step
      nuc_b = np.round(nuc_b / step) * step
    return nuc_a, nuc_b
  
  def process(self, input_data=None, args=None):
    corr = input_data['ADC'] * self.coef_a + self.coef_b
    adc_max = 2**self.adc_resolution - 1
    corr_sat = np.where(corr < adc_max, corr, adc_max)
    return {"ADC" : corr_sat}
  
  def get_parameter_id_str(self, args, input_data):
    return str(args)
  
  def store_display(self, prefix, args, input_data, output_data, cached):
    for key in output_data:
      fname = prefix + key + '.png'
      if not os.path.exists(fname) or not cached:
        output_normalized = 255.0*output_data[key]/(2**self.adc_resolution - 1)
        image_data = output_normalized.astype(np.uint8)
        #image_data = cv2.equalizeHist(image_data)
        image = Image.fromarray(image_data)
        image.save(fname)



 
