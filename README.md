# IR Sensor Non Uniformity Simulator

### Dependencies
Non Uniformity Simulator model is build using:
 - Linux Ubuntu 16
 - Python 3.5.2
 - Numpy 1.18.5
 - Scipy 1.4.1
 - Matplotlib 3.0.3
 - PIL 7.2.0
 - Argparse 1.1

### Consist of
 - sys_param(.py)
 - tolerance(.py)
 - blackbody(.py)
 - frame_gen(.py)
 - one_pix(.py)
 - dpd(.py)
 - nuc(.py)

### Short description
Non Uniformity Simulator consist of four python scripts:
 Non Uniformity Simulator consist of four python scripts:
 - sys_param(.py)
    - Contains psysical, dimentional, electrical and other parameters of the system.
 - tolerance(.py)
    - Generates HOR x VERT arrays of physical parameters, in the way that each pixel has its own physical parameters different from neighbor pixels.
    - "HOR" is amount of pixels in horizontal dimension (default 646) and "VERT" is amount of pixels in vertical dimention (default 486). Both includes "active", "boundary" and "skimming" pixels.
    - The variation of physical parameter is set as standard deviation of normal distribution. Standard deviation value is calculated as a fraction of nominal value of physical parameter. 
    - The fraction value is set as "argparse" argument when running the script. 
    - Nominal values of physical parameters are defined in "sys_param.(py)" module.
 - blackbody(.py)
    - Calculates in-band integral value of Planck's radiation function, at given wave length range and blackbody temperature. 
    - Using in-band integral value of power radiated by blackbody, calculates IR power distribution over sensor sensor area at given temperature of blackbody, and at given field of view. 
    - IR power impinget on sensor area is calculated using view factor and Abbe's sine condition.
    - IR power distribution over sensor area is calculated taking in account "cosine-to-four" effect.
 - frame_gen(.py)
    - Solves numerically microbolometer equation for each pixel (active, boundary and skimming) and calculates the output voltage of each pixel at given input IR power, bias current, sensor temperature, camera internal temperature and physical parameters of pixel, created by running "tolerance(.py)" skript.
    - Simulates ROIC integrator cell for each pixel (active and boundary) and calculates output voltage at given inputs of active or boundary pixel voltage (inverting input) and skimming pixel voltage (non-inverting input).
    - Converts ROIC integrator output voltage to integer values, that corresponds to 14bit ADC
    - Creates images using output data, previosly converting the solutions to "unsigned integer 8" data type.
    - Can use IR power distribution data, previosly created by running "blackbody(.py)" or call this script directly and use data created, with "argparse" arguments.
 - one_pix(.py)
    - Solves microbolometer equation for one active or boundary pixel and one skimming pixel, using nomilal physical values from "sys_param(.py)" module.
    - Output is graphical representation (plots) of ROIC integrator output, at different IR power input, sensor temperature, camera internal temperature and passive components around OP amp
 - dpd(.py)
    - Dead/Defective Pixel Detection ("DPD").
    - Uses "frame_gen(.py)" output data to find dead pixels (stuck High/Low), and defective pixels (abnormal sensitivity)
    - Saves Dead Pixel Map (.txt file) as output, after data processing.
### Usage

Put all four scrips into one directory.

##### sys_param(.py)
In sys_param(.py) you can define physical, dimensional and other parameters for all the system.

----
##### tolerance(.py)
First run the script tolerance(.py) to create arrays with physical parameters for each pixel.
Example:
```sh
$ python tolerance.py 0.0001 0.0001 0.0001
```
To get the information about positional arguments run:
```sh
$ python tolerance.py -h
```
Directory named "tolerance_data" will appear inside the current directory. It should contain arrays with physical parameters in '.txt' format.


> IMPORTANT NOTES 


- 1. This arrays with physical parameters ensures that output images has the same Non Uniformity pattern.
- 2. Every time you run tolerance(.py) script previous shape is lost and new is created.  
- 3. If you change any value of parameters: 'T_sa', 'R_ta_i', 'g_ini', 'c_ini', 'alpha' or 'Ea' in sys_param(.py) you have to rerun tolerance(.py) script!

 ----
##### blackbody(.py)
To get the information about positional arguments run:
```sh
$ python blackbody.py -h
```
Example below calculates and saves data of IR power distribution over sensor area, at each blackbody temperature in range 300 K to 400 K, with incriment step 1 K. 
```sh
$ python blackbody.py -d 300 400 1
```
Data will be saved in 'data_files' directory in '.txt' format.

Example below will show the grafh of IR power dependency on blackbody temperature. In this case the calculated IR power is that impinges on area, equal to one pixel sensetive area and is located in the middle of the sensor.
```sh
$ python blackbody.py -g 300 400 1
```

 ----
##### frame_gen(.py)
 - Example 1:
 ```sh
 $ python frame_gen.py -sr -ff
 ```
Command row above will run simulation of IR sensor using power distribution data from file, previously created by running "blackbody.py", and will use skimming row as reference for integrator OP amp.

- Example 2:
```sh
$ python frame_gen.py -sc -ff
```
Command row above will run simulation of IR sensor using power distribution data from file, previously created by running "blackbody.py", and will use skimming column as reference for integrator OP amp.

- Example 3:
```sh
$ python frame_gen.py -sr -rbb 300 400 50
```
Command row above will run simulation of IR sensor using skimming row as reference for integrator OP amp, and will call "blackbody(.py)" to create IR power distribution data:
```sh
$ pynton blackbody.py -d 300 400 50 -r
```

Output data and images created will be saved in "data_files/dataSet_time_date" directory.