
"""
    Script for reading in map of Br(theta, phi) on the solar surface and computing an outflow field. 
    There are options for saving the field as a netcdf file, either on a staggered grid or averaged to gridpoints
    There is also included a 3D field line plotter, written by Prof. Anthony R. Yeates (Durham University).
    The facility for plotting and saving multiple field with differing parameters is available by increasing 'plot_num'.
    
    
    Copyright (C) Oliver E.K. Rice, Durham University 24/8/21
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#-------------------------------------------------------

import numpy as n
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import bfield
import plot3d as p3d

#-------------------------------------------------------

"""
Set parameters here, to be run through in the plot_num loop
"""

plot_num = 1 #number of parameter sets to plot/output
fig_width = 513.11743/72  # Width of the figures to be plotted

for i in range(plot_num): 
    """
    Running through the different parameter sets and initialising the magnetic field solver
    """
    
    rss = 2.5 #upper boundary of the domain (solar radii). Must be smaller than the critical radius 'rcrit'
    v1 = 50 #outflow velocity (in code units). Proper conversion to km/s is to be determined at a later date. Set to zero to obtain a potential field.
    rcrit = 10.0 #critical radius, in the solar wind speed function. 10 solar radii is a reasonable approximation to reality
    
    # Set grid resolution in each direction. Must be even numbers. 
    nr = 60    #Radial Direction
    ns = 90    #Latitudinal Direction
    np = 180   #Longitudinal Direction
    
    converge_lim = 0.1 #Set to zero to match lower boundary condition to machine precision.
    # If nonzero, this is the acceptable relative error in the lower boundary. 0.1 = 10% for instance.
    # This is calculated by integrating the absolute difference between the imported magnetogram and the calculated lower boundary.
    # Increasing this number increases the speed of the code, and doesn't affect the field greatly away from the lower boundary.

    import_data = True # Imports HMI data. If false, uses the binit function below for a prescribed boundary condition
    crot = 2165 # Which Carrington rotation to import if import_data is true (integer required here)
    
    save_at_gridpts = True # Saves field at gridpoints as a netcdf file
    save_at_staggered = False # Saves field on the staggered grid. This will happen anyway if a 3d plot is required
   
    """
    Plotting options
    """
    
    plot_3d = True # Plots magnetic field lines in 3d, using Anthony Yeates' field line plotter
    ubound = 2 # Start of field line tracing in plot3d. 0=bottom, 1=top, 2=both
    plot_2d = False  # Plots the radial magnetic field at the upper and lower boundaries, and the imported data
    colour = 1 # On the 3d plot; 0 = grayscale with colourful field lines. 1 = sun coloured with white field lines
    
    file_3d = '3dplot.eps' #filename of the 3d plot. Use .eps for vector image, .png if not.
    
    if v1 == 0:
        print('Initialising parameters... Potential Field, v1 = 0')
    else:
        print('Initialising parameters... Outflow Field, v1 = %d' % v1)
        
    # -------------------------------------------------------------------------------------
    
    """
    Read in the parameter set specified above.
    """
    
    bf = bfield.bfield(v1,nr,ns,np,rcrit,rss,converge_lim)  
    
    if import_data:
        # - how much to smooth
        hmi_smooth = 5e-4   # reasonable for 1-degree resolution [try experimenting]
        print('Importing data from Carrington rotation %i...' % crot)
        br_bound = bf.readmap(crot, smooth=hmi_smooth)   #reading in real data, for the carrington rotaion 'crot'
    
    else: 
        def binit(bf):  #Alternatively, prescribe the lower boundary condition here with a given function psi0
            s, p = n.meshgrid(bf.sc[1:-1], bf.pc[1:-1],indexing='ij')
            #psi0 is the lower boundary condition, with dimensions equal to that of s and p above
            psi0 = s**7
            area = bf.Sr[0,0,0]
            print('Divergence inside sphere: ', n.sum(psi0)*area)
            if abs(n.sum(psi0)*area) > 1e-10:
                raise Exception('Lower Condition not Divergence-Free')
            return psi0  #lower boundary condition
        br_bound = binit(bf)  #read in data from binit
    
    print('Lower Boundary Condition Set. Calculating Magnetic Field...')
    
    """
    Boundary data has been imported. Plot it if necessary and calculate the magnetic field using the function bf.findb
    """
    
    if plot_2d:   
        plt.figure(figsize=(7,3))
        plt.pcolormesh(bf.ps*180/n.pi, bf.ss, br_bound, cmap='bwr', vmin=-n.max(n.abs(br_bound)), vmax=n.max(n.abs(br_bound)))
        plt.title('Lower Boundary Magnetogram Data')
        plt.colorbar(label='Radial Magnetic Flux Density (Gauss)')
        plt.xlabel('Carrington Longitude ($\phi$)')
        plt.ylabel(r'Latitude ($\cos \theta$)')
        plt.tight_layout()
        plt.savefig('bfieldimport%d.png' % i)
        plt.close()
        
    br,bs,bp = bf.findb(br_bound)  #calculates the magnetic field on a staggered grid.
    
    print('Field calculated, proceeding...')
    
    if plot_3d:
        save_at_gridpts = True  #to plot the 3d field, we need the field at the gridpoints
        
    """
    To save at the gridpoints, we need to define ghost points and then take averages. By doing this we lose the divergence-free condition. 
    Saving on the staggered grid preserves this condition, but the 3d field line plotter requires the field at grid points.
    """
    
    if save_at_gridpts:
        filename = 'bgrid%d.nc' % i
        brg,bsg,bpg = bf.ghosts(br,bs,bp)  #extends b to give ghost points in the directions on the faces
        br0,bs0,bp0 = bf.b_to_gridpts(brg,bsg,bpg)   #averages b to gridpoints. The field is exported in this format by default.
        bf.bggrid(filename,n.exp(bf.rs),n.arccos(bf.ss),bf.ps,n.swapaxes(br0,0,2),n.swapaxes(bs0,0,2),n.swapaxes(bp0,0,2))  #saves field
    
    if save_at_staggered:
        filename = 'bstagger%d.nc' % i
        bf.bgstagger(filename,n.swapaxes(br,0,2),n.swapaxes(bs,0,2),n.swapaxes(bp,0,2))  #saves field
    
    if plot_2d:  #plots the calculated magnetic field at the upper and lower boundaries
        plt.figure(figsize=(7,3))
        plt.pcolormesh(bf.ps*180/n.pi, bf.ss, br[0], cmap='bwr', vmin=-n.max(n.abs(br_bound)), vmax=n.max(n.abs(br_bound)))
        plt.title('Lower Boundary Magnetic Field')
        plt.colorbar(label='Radial Magnetic Flux Density (Gauss)')
        plt.xlabel('Carrington Longitude ($\phi$)')
        plt.ylabel(r'Latitude ($\cos \theta$)')
        plt.tight_layout()
        plt.savefig('bfieldlower%d.png' % i)
        plt.close()
        
        plt.figure(figsize=(7,3))
        plt.pcolormesh(bf.ps*180/n.pi, bf.ss, br[-1], cmap='bwr', vmin=-n.max(n.abs(br[-1])), vmax=n.max(n.abs(br[-1])))
        plt.title('Upper Boundary Magnetic Field')
        plt.colorbar(label='Radial Magnetic Flux Density (Gauss)')
        plt.xlabel('Carrington Longitude ($\phi$)')
        plt.ylabel(r'Latitude ($\cos \theta$)')
        plt.tight_layout()
        plt.savefig('bfieldupper%d.png' % i)
        plt.close()
                      
if plot_3d:  #plot the 3d field
    p3d.plot3d(bf.rss,ubound,file_3d,plot_num,colour)
        
