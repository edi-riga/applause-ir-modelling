Simulation is made from one or several simulation model instances:
```python
blackbody = Blackbody()
```

Each model may have configuration parameters that are set during instattiation and remain constant throughout the simulation. If the `vusualize` option is set to `True` the model will save the intermediate results in graphics format.
```python
optics = Optics(resolution=(640, 480), visualize=True)
```

Some models may also have an additional argument. It is possible to supply a list of argument values, this will split the simulation to several branches.
```python
blackbody.set_args_list([300, 400])
```

Simulation constructor accepts a list of models. Each model processes the data received from the previous model and passes the result to the next one. The first model in the list is different in that it can not accept any input data and must instead generate it by itself.
```python
sim = Simulation([blackbody, optics])
output = sim.process()
```

The above example simulates the black body radiator at temperatures of 300K and 400K, then models the effect of the lens on both images. The results are stored in the `output` array and as images in the `display` directory.
