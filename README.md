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

### Short description
Non Uniformity Simulator consist of four python scripts:
 - sys_param(.py)
Contains psysical, dimentional, and other parameters of the system.
 - tolerance(.py)
 Generates 640x480 arrays of physical parameters, in the way that each pixel has its own physical parameters different from neighbor pixels. The variation of physical parameter is set as standard deviation of normal distribution. Standard deviation value is calculated as a fraction of nominal value of physical parameter. The fraction value is set as "argparse" argument when running the script.

 - blackbody(.py)
 Calculates IR power impinged on each pixel of the sensor at given temperature of blackbody.
 Blackbody radiation is calculated by in-band integration of Planck's radiation function, at given wave length range.
 IR power impinget on sensor area is calculated using view factor and Abbe's sine condition.
 IR power distribution over sensor area is calculated taking in account "cosine-to-four" effect.
 - frame_gen(.py)
 Calls the script blackbody(.py) in Linux terminal.
 Calculates each pixel output voltage at given IR power, blackbody temperature range and sensor temperature.
 Generates images that correspond to given blackbody and sensor temperatures.

### Usage

Put all four scrips into one directory.
Scripts should be run from Linux terminal.
##### sys_param(.py)
In sys_param(.py) you can define physical, dimensional and other parameters for all the system.
##### tolerance(.py)
First run the script tolerance(.py) to create arrays with physical parameters for each pixel.
Example:
```sh
$ python3 tolerance.py 0.0001 0.0001 0.0001
```
To get the information about positional arguments run:
```sh
$ python3 tolerance.py -h
```
Directory named "tolerance_data" will appear inside the current directory. It should contain arrays with physical parameters in '.txt' format.
| IMPORTANT NOTES |
| --- |
| 1. This arrays with physical parameters ensures that outut images has the same Non Uniformity pattern.  |
| 2. Every time you run tolerance(.py) script previous shape is lost and new is created.    |
| 3. If you change any value of parameters: 'T_sa', 'R_ta_i', 'g_ini', 'c_ini', 'alpha' or 'Ea' in sys_param(.py) you have to rerun tolerance(.py) script! |

##### blackbody(.py)
To get the information about positional arguments run:
```sh
$ python3 blackbody.py -h
```
Example below calculates and saves data of IR power distribution over sensor area, at each blackbody temperature in range 300 K to 400 K, with incriment step 1 K. 
```sh
$ python3 blackbody.py -d 300 400 1
```
Data will be saved in 'data_files' directory in '.txt' format.

Example below will show the grafh of IR power dependency on blackbody temperature. In this case the calculated IR power is that impinges on area, equal to one pixel sensetive area and is located in the middle of the sensor.
```sh
$ python3 blackbody.py -g 300 400 1
```

##### frame_gen(.py)
 - Example 1:
 ```sh
 $ python3 frame_gen.py -b 300 400 1 300
 ```
Comand row above will run:
```sh
$ python3 blackdody.py -d 300 400 1
```
and then will use calculated IR power distribution data to get each pixel output voltage at sensor ampient temperature equal 300 K (the last argument in first command row).

- Example 2:
```sh
$ python3 frame_gen.py -s 300 350 1 350
```
Command row above will run:
```sh
$ pynton3 blackbody.py -d 350 350 1
```
and will get data for IR power distribution at blackbody temperature 350 K (the last argument in firs command row).
Then it will calculate each pixel output voltage at given IR power distribution, at each sensor ambient temperature in range 300 K to 350 K with incremental step equal to 1 K.

Calculated data will be saved in the directory 'data_files'.
Images will be saved in the directory 'frames'.