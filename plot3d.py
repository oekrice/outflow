
"""
    Script for reading magnetic field from netcdf file, tracing some magnetic field lines, and plotting the result in the plane-of-sky.
    
    Note that the netcdf file must be of grid point type (i.e. br, bth, bph all
    co-located at the mesh points).
    
    Original plotter Copyright (C) Anthony R. Yeates, Durham University 21/9/17,
    modified version by Oliver E.K. Rice, Durham University 24/8/21.
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

import numpy as n
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.integrate import ode
from sunpy.coordinates.sun import  L0,B0
from datetime import datetime
#3D outflow code in spherical coordinates

"""
Calculate latitude and longitude for a given date, alternatively just specify these angles
"""

def view_angle(viewdate): 
    clon = L0(time=viewdate)
    clat = B0(time=viewdate)
    lon0 = n.deg2rad(clon.dms[0])
    lat0 = n.deg2rad(clat.dms[0])
    return lon0, lat0

fig_width = 513.11743/72  #Latex text width. 
#fig_width = 60
"""
The below function contains everything to do the plotting, and is designed to plot multiple images on the same Figure, if plot_num>1
"""
    
def plot3d(r1,line_starts,file_3d,plot_num,colour):  
    fig3d = plt.figure(constrained_layout = True)
    plt.figure(figsize = (fig_width,fig_width))
    """
    Arrange plots on the page
    """
    if n.abs(n.sqrt(plot_num) - int(n.sqrt(plot_num))) < 1e-10:  #Perfect square of figures
        plot_size = int(n.sqrt(plot_num))
    else:
        plot_size = int(n.sqrt(plot_num))+1
    gs = fig3d.add_gridspec(plot_size,plot_size)
    
    # ------------------------------------------------------------------
    for i in range(plot_num):
        bfile = 'bgrid%d.nc' % i
        ndegs = 360
        degs = n.linspace(0,360,ndegs+1)
        degs = [0]
        for deg1 in range(len(degs)):
            # SPECIFY DIRECTION OF VIEWER:
            time = datetime(2012, 11, 13, 12)
            lon0, lat0 = view_angle(time)
            print(lon0,lat0)
            # SPECIFY MAXIMUM OF COLOUR SCALE FOR Br ON SOLAR SURFACE:
            bmax = 20.0
    
            # SPECIFY FIELD LINE START POINTS:
            # - these should be 1d arrays in theta, phi and r (usual spherical coordinates):
            # - Change the length of the linspace arrays (nth and nph) to change the numer of field lines plotted in each direction
            
            if line_starts == 1:  #trace from top
                nth = 16
                nph = 12
                th00 = n.linspace(n.pi/8, 7*n.pi/8, nth)
                th0 = n.tile(th00, nph)
                ph00 = n.linspace(180/nph, 2*n.pi-180/nph, nph)
                ph0 = n.repeat(ph00,nth)
                r0 = th0*0 + r1
            if line_starts == 0:  #trace from bottom
                nth = 32
                nph = 24
                th00 = n.linspace(n.pi/8, 7*n.pi/8, nth)
                th0 = n.tile(th00, nph)
                ph00 = n.linspace(180/nph, 2*n.pi-180/nph, nph)
                ph0 = n.repeat(ph00,nth)
                #r0 = th0*0 + 1.0
                r0 = th0*0 + 1.0
            
            if line_starts == 2:  #trace both, removing some of them to avoid clutter
                #Upper Boundary
                if colour == 0:
                    nth = 32
                    nph = 2
                else:
                    nth = 90
                    nph = 2
                th00 = n.linspace(n.pi/nth, n.pi-n.pi/nth, nth)
                th01 = n.tile(th00, nph)
                ph00 = n.array([lon0 - n.pi/2, lon0 + n.pi/2])
                ph01 = n.repeat(ph00,nth)
                r01 = th01*0 + r1
                #Lower Boundary
                if colour == 0:
                    nth = 18
                    nph = 18
                else:
                    nth = 36
                    nph = 36
                th00 = n.linspace(n.pi/nth, n.pi-n.pi/nth, nth)
                th02 = n.tile(th00, nph)
                ph00 = n.linspace(n.pi/nph, 2*n.pi-n.pi/nph, nph)
                ph02 = n.repeat(ph00,nth)
                r02 = th02*0 + 1.0
                #combine the above start points
                th0 = n.concatenate((th01,th02),axis=None)
                ph0 = n.concatenate((ph01,ph02),axis=None)
                r0 = n.concatenate((r01,r02),axis=None)
            #-----------------------------------------------------------------------
            def bTrace(t, x):
                """
                Return B/|B| for use by the field line tracer.
                """
                # (ph,s,rh) coordinates of current point:
                ph = (n.arctan2(x[1], x[0]) + 2*n.pi) % (2*n.pi)
                r = n.sqrt(n.sum(x**2))
                s = x[2]/r
                rh = n.log(r)
                b1 = brgi( n.stack((ph, s, rh)) )
                return b1/n.linalg.norm(b1)
    
            def trace(x0, dtf=1e-2, tol=1e-2, nrefine=3):
                """
                Trace the fieldline starting from x0, using scipy.integrate.ode.
                - uses implicit Adams method (up to order 12).
                - the parameter dtf is the maximum step-size, which
                will be the output resolution of the field line in most of the domain.
                - the tolerance for tracing is tol*dt.  
                - nrefine is the number of times to refine the step-size to get close
                to the boundary.
                """
                xl = x0.copy()
    
                # Backwards:
                t = 0.0
                dt = dtf
                for j in range(nrefine):
                    solver = ode(bTrace).set_integrator('vode', method='adams', atol=tol*dt)
                    solver.set_initial_value(xl[:,0:1], t)
                    while True:
                        try:
                            solver.integrate(solver.t - dt)
                            xl = n.insert(xl, [0], solver.y, axis=1)
                        except ValueError: # reached boundary
                            break
                    t = solver.t
                    dt /= 10.0
    
                # Forwards:
                t = 0.0
                dt = dtf
                for j in range(nrefine):
                    solver = ode(bTrace).set_integrator('vode', method='adams', atol=tol*dt)
                    solver.set_initial_value(xl[:,-1:], t)
                    while True:
                        try:
                            solver.integrate(solver.t + dt)
                            xl = n.append(xl, solver.y, axis=1)
                        except ValueError: # reached boundary
                            break
                    t = solver.t
                    dt /= 10.0
                return xl[0,:], xl[1,:], xl[2,:]
    
            
            #-----------------------------------------------------------------------
            # COMPUTE FIELD LINES:
            # - read in magnetic field:
            fh = netcdf.netcdf_file(bfile, 'r', mmap=False)
            r = fh.variables['r'][:]
            th = fh.variables['th'][:]
            ph = fh.variables['ph'][:]
            br = fh.variables['br'][:]
            bth = fh.variables['bth'][:]
            bph = fh.variables['bph'][:]
            fh.close()
    
            # - (rho,s,phi) coordinates:
            rh = n.log(r)
            s = n.cos(th)
            # - convert to Cartesian components and make interpolator on (rho,s,phi) grid:
            ph3, s3, rh3 = n.meshgrid(ph, s, rh, indexing='ij')
            bx = n.sqrt(1-s3**2)*n.cos(ph3)*br + s3*n.cos(ph3)*bth - n.sin(ph3)*bph
            by = n.sqrt(1-s3**2)*n.sin(ph3)*br + s3*n.sin(ph3)*bth + n.cos(ph3)*bph
            bz = s3*br - n.sqrt(1-s3**2)*bth
            del(br, bth, bph)
            bstack = n.stack((bx,by,bz),axis=3)
            del(bx, by, bz)
            brgi = rgi((ph, s, rh), bstack)
            del(bstack)
            # - convert starting points to Cartesian coordinates:
            x0 = n.stack((r0*n.cos(ph0)*n.sin(th0), r0*n.sin(ph0)*n.sin(th0), r0*n.cos(th0)), axis=0)
    
            # MAKE PROJECTION OF Br ON SOLAR SURFACE:
            fh = netcdf.netcdf_file(bfile, 'r', mmap=False)
            th = fh.variables['th'][:]
            ph = fh.variables['ph'][:]
            br = fh.variables['br'][:][:,:,0]
            fh.close()
            # - make regular grid interpolator:
            s = n.cos(th)
            br2 = rgi((ph,s), br, bounds_error=False, fill_value=-bmax)   
            # - project on to sphere:
            xx = n.linspace(-1, 1, 256)
            y2, z2 = n.meshgrid(xx, xx, indexing='ij')
            r2 = n.sqrt(y2**2 + z2**2)
            x2 = y2*0
            x2[r2 <= 1] = n.sqrt(1.0 - r2[r2 <= 1]**2)
            x2, z2 = n.cos(lat0)*x2 - n.sin(lat0)*z2, n.sin(lat0)*x2 + n.cos(lat0)*z2
            x2, y2 = n.cos(lon0)*x2 - n.sin(lon0)*y2, n.sin(lon0)*x2 + n.cos(lon0)*y2    
            x2[r2 > 1] = 0
            y2[r2 > 1] = 0
            sp = z2    
            php = (n.arctan2(y2, x2) + 2*n.pi) % (2*n.pi)
            brp = br2(n.stack((php, sp), axis=2))
            brp[r2 > 1] = -bmax
                
            # PLOT:
            # - set up figure:
            #plt.figure(figsize=(12,12))
            #ax = plt.subplot(111)
            ax = plt.subplot(gs[i//plot_size,i%plot_size])
            #ax = plt.subplot(gs[1,1])
            ax.set_facecolor('k')
            if colour == 0:
                cmap0 = plt.cm.get_cmap('gray')
            else:
                cmap0 = plt.cm.get_cmap('hot')
            # - plot br on the solar surface:
            lev = n.linspace(-bmax, bmax, 128)
            y2, z2 = n.meshgrid(xx, xx, indexing='ij')  
            plt.contourf(y2, z2, brp, lev,rasterized=True,cmap=cmap0, extend='both')
            # - trace and plot field lines:
            nl = n.size(r0)
            for j in range(nl):
                xl, yl, zl = trace(x0[:,j:j+1]) 
                #   - rotate to correct viewing direction:
                xl, yl = xl*n.cos(lon0) + yl*n.sin(lon0), -xl*n.sin(lon0) + yl*n.cos(lon0)
                xl, zl = xl*n.cos(lat0) + zl*n.sin(lat0), -xl*n.sin(lat0) + zl*n.cos(lat0)
                #   - remove points that are behind the Sun:
                rl = n.sqrt(yl**2 + zl**2)
                ind = (rl >= 1) | (xl > 0)
                #   - plot this line:
                if line_starts == 2:
                    #don't plot lines that come from the surface and end up at the top
                    startr = n.sqrt(xl[0]**2 + yl[0]**2 + zl[0]**2)
                    endr = n.sqrt(xl[-1]**2 + yl[-1]**2 + zl[-1]**2)
                    x0abs = n.sqrt(x0[0,j:j+1]**2 + x0[1,j:j+1]**2 + x0[2,j:j+1]**2)
                    if abs(startr - endr) < 0.1:
                        if colour == 0:
                            ax.plot(yl[ind], zl[ind], linewidth=0.1)#, c='white')
                        else:
                            ax.plot(yl[ind], zl[ind], linewidth=0.01, c='white')
                    elif x0abs > r1-0.1:
                        if colour == 0:
                            ax.plot(yl[ind], zl[ind], linewidth=0.1)#, c='white')
                        else:
                            ax.plot(yl[ind], zl[ind], linewidth=0.01, c='white')
                else:
                    ax.plot(yl[ind], zl[ind], linewidth=0.1)#, c='white')
            # - tidy plot window:
            rmax = r1
            ax.set_xlim(-rmax, rmax)
            ax.set_ylim(-rmax, rmax)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', \
                            left='off', right='off', labelbottom='off', \
                            labelleft='off')
            plt.tight_layout

    plt.savefig(file_3d)
    plt.close()

#plot3d(2.5,2,'3dplot.eps',1,1)
