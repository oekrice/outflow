# outflow
Python script to calculate equilibrium solutions of the magnetofrictional model, taking into account solar wind outflow in the solar corona. 

'runoutflow.py' is a shell script, within which the code is initialised. All the mathematical functions for calculating the magnetic field and importing data from HMI are contained within 'bfield.py'. The chosen function for the solar wind profile is set in bfield (the function vout).

There is an option to plot multiple 3d plots of the calculated magnetic field. This uses a field line tracer and plotter written by Anthony Yeates, within the file 'plot3d.py'. Options for plotting 2d fields are also given.

For more information on the mathematical methods involved, please see our future paper...

