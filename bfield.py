
"""
        Compute 3D outflow field using eigenfunction method in r,s,p coordinates, on the dumfric grid (equally spaced in
        rho=ln(r/rsun), s=cos(theta0), and p=phi).
       
        The output should have zero divergence to machine precision, on a staggered grid

        Parameters are set in the shell script 'runoutflow.py'. This script contains the required functions to read in HMI data and calculate the field.
        
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

# - Import modules

import numpy as n
from scipy.io import netcdf
from scipy.linalg import eigh_tridiagonal
from astropy.io import fits
from scipy.interpolate import interp2d
import drms

"""
Initialise the class containing the parameter data (outflow velocity, resolutions etc.)
"""

class bfield:   
    def __init__(self,v1,nr,ns,np,rcrit,rss,converge_lim):
        self.v1 = v1
        self.nr = nr
        self.ns = ns
        self.np = np
        self.rcrit = rcrit
        self.rss = rss
        self.r0, self.r1 = n.log(1.0), n.log(rss) #radius limit
        self.s0, self.s1 = -1,1       #theta limit
        self.p0, self.p1 = 0,2*n.pi   #phi limit
        #calculate step sizes
        self.dr = (self.r1-self.r0)/self.nr  
        self.ds = (self.s1-self.s0)/self.ns
        self.dp = (self.p1-self.p0)/self.np
        #coordinate axes on gridpoints
        self.rs = n.linspace(self.r0,self.r1,nr+1)  
        self.ss = n.linspace(self.s0,self.s1,ns+1)
        self.ps = n.linspace(self.p0,self.p1,np+1)
        #coordinate axes on grid faces
        self.rc = n.linspace(self.r0-self.dr/2,self.r1 + self.dr/2,self.nr+2) 
        self.sc = n.linspace(self.s0-self.ds/2,self.s1 + self.ds/2,self.ns+2)
        self.sc[0] = self.sc[1]; self.sc[-1] = self.sc[-2]
        self.pc = n.linspace(self.p0-self.dp/2,self.p1 + self.dp/2,self.np+2)
        
        self.converge_lim = converge_lim
        
        def areas(self): #returns the areas of faces as a 3d array
            r,s,p = n.meshgrid(self.rs,self.ss,self.ps,indexing='ij')
            Sr = n.exp(2*r[:,1:,1:])*self.ds*self.dp
            Ss = 0.5 * (n.exp(2*r[1:,:,1:]) - n.exp(2*r[:-1,:,1:])) * n.sqrt(n.ones((self.nr,self.ns+1,self.np))-s[:-1,:,1:]**2) * self.dp
            Sp = 0.5 * (n.exp(2*r[1:,1:,:]) - n.exp(2*r[:-1,1:,:])) * (n.arcsin(s[1:,1:,:])-n.arcsin(s[1:,:-1,:]))
            return Sr, Ss, Sp

        self.Sr, self.Ss, self.Sp = areas(self)
        
        def volume(self): #used only in the energy calculation. The volume of each grid 'cube' is actually the same so this isn't necessary. But if a different coordinates system is used this can be modified.
            r,s,p = n.meshgrid(self.rs,self.sc[1:-1],self.pc[1:-1],indexing='ij')   #3d r box
            V = (4/3)*(n.exp(3*r[1:]) - n.exp(3*r[:-1]))*self.ds*self.dp
            return V
    
        self.V = volume(self)    
    
    """
    Calculate the eigenvalues and eigenvectors in the azimuthal direction. The eigenvales are integers in the infinite limit but are not necessarily so here.
    """
    
    def findms(self): 
        # - We have to combine the eigenvectors from both the cosines and the sines, hence this is more complicated than the equivalent in the latitudinal direction.
        num = len(self.pc[1:-1])
        dvals = 2*n.ones((num))
        evals = -n.ones((num-1))
        dvals[0] = 1; dvals[-1] = 1
        w1,v1 = eigh_tridiagonal(dvals,evals)  #calculate sine eigenvalues
        ms1 = []
        for i in range(0,len(w1)): 
            ms1.append(n.sqrt(abs(w1[i])/self.dp**2))
        dvals[0] = 3; dvals[-1] = 3
        w2,v2 = eigh_tridiagonal(dvals,evals)  #calculate cosine eigenvalues
        ms2 = []
        for i in range(0,len(w2)): 
            ms2.append(n.sqrt(abs(w2[i])/self.dp**2))
        v = v1*0
        ms = ms1*0 
        for i in range(len(ms1)):  #combine the two sets of eigenvalues as necessary, discarding the ones correspoinding to half-integers (which do not satisfy the boundary condition)
            if i%2 == 0:
                ms.append(ms1[i])
                v[:,i] = v1[:,i]
            else:
                ms.append(ms2[i])
                v[:,i] = v2[:,i]
        return ms, v

    """
    Calculate eigenvalues and eigenvectors in the latitudinal direction
    """

    def findls(self,m):
        sigc = n.sqrt(1-self.sc**2) #sigma evaluated at the cell midpoints
        sigs = n.sqrt(1-self.ss**2)
        evals = -sigs[1:-1]**2  
        dvals = sigs[1:]**2 + sigs[:-1]**2 - (self.ds*m**2/sigc[1:-1])*(n.arcsin(self.ss[:-1])-n.arcsin(self.ss[1:]))
        w,v = eigh_tridiagonal(dvals,evals)
        return w/self.ds**2, v
    
    """ 
    Find the coefficients cmn in order to match the lower boundary condition
    """
    
    def coeff(self, br0, l2,m,q,p):  
        Sc = n.tile(q,(self.np,1)).T  
        Pc1 = n.tile(p,(self.ns,1))
        lhs = n.sum(br0*Sc*Pc1)
        rhs = n.sum(Sc*Sc*Pc1*Pc1)
        return lhs/rhs

    """
    Solar wind function is specified here, along with its analytic derivative
    """
    
    def vout(self,r):  
        top = n.exp(-2*self.rcrit/n.exp(r))*n.exp(self.r1*2)
        bottom = n.exp(-2*self.rcrit/n.exp(self.r1))*n.exp(r*2)
        return self.v1*top/bottom    # This is the Parker solar wind solution, but it can be changed to any reasonable equivalent.
    
    def voutdiff(self,r): #the analytical derivative of the solar wind solution, similarly to above
        A = self.v1 * self.rss ** 2 * n.exp(2.0*self.rcrit/self.rss)
        return n.exp(r) * A * 2 * n.exp(-2*self.rcrit/n.exp(r)) * (self.rcrit - n.exp(r)) * n.exp(r)**-4

    """
    Calculate the required functions in the radial direction. Not an eigenvalue problem, unlike in the other directions.
    """

    def findh(self,l):  #finds the H function, normalised to satisfy the lower boundary condition.
        #The exact calculation here shouldn't affect the solenoidal condition on B (that is ensured when calculating Q).
        hc = self.rc*0  #initialise grid
        hc[-1] = 0; hc[-1] = -1 #This isn't strictly accurate but seems to be the best numerical option.
        def findvs(self):  #finds staggered v_out, unstaggered v_out and the appropriate derivative
            vc = self.rc*0
            vs = self.rs*0
            vd = self.rc*0
            for i in range(len(self.rc)):
                vc[i] = self.vout(self.rc[i])
                vd[i] = self.voutdiff(self.rc[i])
            for i in range(len(self.rs)):
                vs[i] = self.vout(self.rs[i])
            return vs, vc, vd
        vs, vc, vd = findvs(self)
        for i in range(len(self.rc) - 3, -1, -1): #work backwards from the top boundary
            #Calculate quantities A, B, C according to scheme described in the code
            A = 1
            B = 3 - vc[i+1]*n.exp(self.rc[i+1])
            C = 2 - l - 3*vc[i+1]*n.exp(self.rc[i+1]) - vd[i+1]*n.exp(self.rc[i+1])
            top = hc[i+1] * (2*A/(self.dr**2) - C) + hc[i+2] * (-A/(self.dr**2) - B/(2*self.dr))
            bottom = (A/(self.dr**2) - B/(2*self.dr))
            hc[i] = top/bottom
        grad = (hc[1]*n.exp(self.rc[1]) - hc[0]*n.exp(self.rc[0]))/self.dr  #to ensure the lower boundary condition is correct.
        return hc/grad

    def findg(self,hc,l): #finds the G function from H using the specified scheme from the notes.
        gs = self.rs*0
        gs[0] = 1  #lower boundary condition on G
        for i in range(1,len(gs)):
            gs[i] = n.exp(-2*self.rs[i])*(0.5*l*hc[i]*(n.exp(2*self.rs[i]) - n.exp(2*self.rs[i-1])) + gs[i-1]*n.exp(2*self.rs[i-1]))
        return gs

    """
    Function to calculate the magnetic field, using all of the above functions.
    """
    
    def findb(self,br0): 
        converge = self.converge_lim
        #obtain blank B fields with the correct dimensions. No ghost points etc. (for now)
        br = n.zeros((self.nr+1,self.ns,self.np))
        bs = n.zeros((self.nr,self.ns+1,self.np))
        bp = n.zeros((self.nr,self.ns,self.np+1))
        ms, trigs = (self.findms())
        ls= n.zeros((len(ms),len(self.ss)-1))
        legs = n.zeros((len(ms),self.ns,self.ns))
        for i in range(len(ms)):
            ls[i],legs[i]  = self.findls(ms[i])
        print('Eigenvalues and vectors calculated')
        cml = ls*0 #Fourier coefficient matrices
        sigs = n.sqrt(n.ones(self.ns+1) - self.ss**2)
        sigc = n.sqrt(n.ones(self.ns) - self.sc[1:-1]**2)
        print('Doing Fourier/Legendre Transform and computing radial functions...')
        check = n.zeros((self.ns+2,self.np+2))
        ijmax = len(ls) + len(ls[i])
        count = 0
        for k in range(ijmax):  #run through the m modes
            for i in range(max(0,k-len(ls[i])+1),min(k+1,len(ls))):  #and the ls for each m
                count += 1
                j = k-i
                cml[i][j] = self.coeff(br0, ls[i][j],ms[i],legs[i,:,j],trigs[:,i])  #Calculate boundary coefficients
                if abs(cml[i][j]) < 1e-10:
                    cml[i][j] = 0
                else:  
                    q = n.zeros((self.ns+2))
                    q[1:-1] = legs[i,:,j]  # Legendre function
                    p=n.zeros((self.np+2))
                    p[1:-1] = trigs[:,i]     # Trig functions
                    p[0] = p[1]; p[-1] = p[-2]
                    h = self.findh(ls[i][j])     # Radial functions
                    g = self.findg(h,ls[i][j])
                    #Then add on each mode to the magnetic fields, differentiating as appropriate. Tiles and transposes deal with dimensions fairly quickly
                    br += cml[i][j] * n.transpose(n.tile(g,(self.ns,self.np,1)),(2,0,1)) * n.transpose(n.tile(q[1:-1],(self.nr+1,self.np,1)),(0,2,1)) * n.transpose(n.tile(p[1:-1],(self.nr+1,self.ns,1)),(0,1,2))
                    bs += cml[i][j] * n.transpose(n.tile(sigs,(self.nr,self.np,1)),(0,2,1)) * n.transpose(n.tile(h[1:-1],(self.ns+1,self.np,1)),(2,0,1)) * n.transpose(n.tile((q[1:]-q[:-1])/self.ds,(self.nr,self.np,1)),(0,2,1)) * n.transpose(n.tile(p[1:-1],(self.nr,self.ns+1,1)),(0,1,2))
                    bp += cml[i][j] * n.transpose(n.tile(1/sigc,(self.nr,self.np+1,1)),(0,2,1)) * n.transpose(n.tile(h[1:-1],(self.ns,self.np+1,1)),(2,0,1)) * n.transpose(n.tile(q[1:-1],(self.nr,self.np+1,1)),(0,2,1)) * n.transpose(n.tile((p[1:]-p[:-1])/self.dp,(self.nr,self.ns,1)),(0,1,2))  
                    Q = n.tile(q,(self.np+2,1)).T #these functions are not used in the resulting field. They are just to check convergence.
                    P = n.tile(p,(self.ns+2,1))
                    check += Q*P*cml[i][j]   #calculating the lower boundary after each mode, to check against the target
                    pcerror = 100.0*n.sum(n.abs(check[1:-1,1:-1]-br0))/n.sum(n.abs(br0))
                    oferror = n.abs(100.0*(n.sum(n.abs(check[1:-1,1:-1]))-n.sum(n.abs(br0)))/n.sum(n.abs(br0)))
                    print('Calculating... Lower Boundary Absolute Error: %5.2f%%, Approx Max. Outflux Error: %5.2f%%, Modes calculated: %d/%d' % (pcerror,oferror,count,self.ns*self.np), end='\r')
                    if pcerror/100 < converge:
                        print('Calculating... Lower Boundary Error: %5.2f%%, Approx Max. Outflux Error: %5.2f%%, Modes calculated: %d/%d' % (pcerror,oferror,count,self.ns*self.np))
                        return br,bs,bp    
        print('Calculating... Lower Boundary Error: %5.2f%%, Approx Max. Outflux Error: %5.2f%%, Modes calculated: %d/%d' % (pcerror,oferror,count,self.ns*self.np))
        return br,bs,bp
    
    
    def outflux(self,br):  #outputs the total absolute outflux. Integrates the modulus of the radial field over the outer area
        return n.sum(abs(br[-1])*self.Sr[-1])
    
    def energy(self,br1,bs1,bp1):
        return n.sum(0.5*self.V*(br1**2 + bs1**2 + bp1**2))
     
    #--------------------------------------------------------------------------------
    
    #The following functions read in HMI data, for use as the lower boundary condition.
    
    def readmap(self,rot, smooth=0):
        """
            Reads the synoptic map for Carrington rotation rot, corrects the flux, and maps to the DuMFric grid.
            Also reads in the neighbouring maps, and puts them together for smoothing.
            
            ARGUMENTS:
                rot is the number of the required Carrington rotation (e.g. 2190)
                ns and nph define the required grid (e.g. 180 and 360)
                smooth [optional] controls the strength of smoothing (default 0 is no smoothing)
            
            [Important: the output map is corrected for flux balance, but the same correction is applied to the
            two neighbouring maps (for continuity), so they are not balanced.]
        """
        nph = self.np
        ns = self.ns
        # (1) READ IN DATA AND STITCH TOGETHER 3 ROTATIONS
        # ------------------------------------------------
        # Read in map and neighbours:
        # The seg='Mr_polfil' downloads the polar field corrected data --> without NaN values and errors due to projection effect
        # Read "Polar Field Correction for HMI Line-of-Sight Synoptic Data" by Xudong Sun, 2018 to know more 
        # Link: https://arxiv.org/pdf/1801.04265.pdf    
        try:
            c = drms.Client()
            seg = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % rot), seg='Mr_polfil')
            segr = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (rot-1)), seg='Mr_polfil')
            segl = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (rot+1)), seg='Mr_polfil')       
        except:
            print('Error downloading HMI synoptic maps -- required rotations: %4.4i, %4.4i, %4.4i' % (rot-1, rot, rot+1))
        
        # Extract data arrays: (data is stored in 2nd slot with No. 1 and not in the PRIMARY (No. 0), thus [1].data)
        # typical file structure:
        # No.    Name      Ver    Type        Cards   Dimensions     Format
        # 0     PRIMARY    1   PrimaryHDU       6     ()      
        # 1                1   CompImageHDU     13    (3600, 1440)   int32
        print(seg.Mr_polfil[0], segr.Mr_polfil[0], segl.Mr_polfil[0])
        with fits.open('http://jsoc.stanford.edu' + seg.Mr_polfil[0]) as fid:
            brm = fid[1].data
        with fits.open('http://jsoc.stanford.edu' + segl.Mr_polfil[0]) as fid:
            brm_l = fid[1].data
        with fits.open('http://jsoc.stanford.edu' + segr.Mr_polfil[0]) as fid:
            brm_r = fid[1].data
    
        # Stitch together:
        brm3 = n.concatenate((brm_l, brm, brm_r), axis=1)
        del(brm, brm_l, brm_r)
        
        # Remove NaNs:
        #brm3 = np.nan_to_num(brm3)
        
        # Coordinates of original map (pretend it goes only once around Sun in longitude!):
        nsm = n.size(brm3, axis=0)
        npm = n.size(brm3, axis=1)
        dsm = 2.0/nsm
        dpm = 2*n.pi/npm
        scm = n.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)  
        pcm = n.linspace(0.5*dpm, 2*n.pi - 0.5*dpm, npm)  
    
        def correct_flux_multiplicative(f):
            """
                Correct the flux balance in the map f (assumes that cells have equal area).
            """
            # Compute positive and negative fluxes:
            ipos = f > 0
            ineg = f < 0
            fluxp = n.abs(n.sum(f[ipos]))
            fluxn = n.abs(n.sum(f[ineg]))
            
            # Rescale both polarities to mean:
            fluxmn = 0.5*(fluxn + fluxp)
            f1 = f.copy()
            f1[ineg] *= fluxmn/fluxn
            f1[ipos] *= fluxmn/fluxp
            
            return f1

        def plgndr(m,x,lmax):
            """
                Evaluate associated Legendre polynomials P_lm(x) for given (positive)
                m, from l=0,lmax, with spherical harmonic normalization included.
                Only elements l=m:lmax are non-zero.
                
                Similar to scipy.special.lpmv except that function only works for
                small l due to overflow, because it doesn't include the normalization.
            """
            
            nx = n.size(x)
            plm = n.zeros((nx, lmax+1))
            pmm = 1
            if (m > 0):
                somx2 = (1-x)*(1+x)
                fact = 1.0
                for i in range(1,m+1):
                    pmm *= somx2*fact/(fact+1)
                    fact += 2
            
            pmm = n.sqrt((m + 0.5)*pmm)
            pmm *= (-1)**m
            plm[:,m] = pmm
            if (m < lmax):
                pmmp1 = x*n.sqrt(2*m + 3)*pmm
                plm[:,m+1] = pmmp1
                if (m < lmax-1):
                    for l in range(m+2,lmax+1):
                        fact1 = n.sqrt(((l-1.0)**2 - m**2)/(4.0*(l-1.0)**2-1.0))
                        fact = n.sqrt((4.0*l**2-1.0)/(l**2-m**2))
                        pll = (x*pmmp1 - pmm*fact1)*fact
                        pmm = pmmp1
                        pmmp1 = pll
                        plm[:,l] = pll
            return plm

        # (2) SMOOTH COMBINED MAP WITH SPHERICAL HARMONIC FILTER
        # ------------------------------------------------------
        if (smooth > 0):
            # Azimuthal dependence by FFT:
            brm3 = n.fft.fft(brm3, axis=1)
    
            # Compute Legendre polynomials on equal (s, ph) grid,
            # with spherical harmonic normalisation:
            lmax = 2*int((nph-1)/2)  # note - already lower resolution
            nm = 2*lmax+1  # only need to compute this many values
            plm = n.zeros((nsm, nm, lmax+1))
            for m in range(lmax+1):
                plm[:,m,:] = plgndr(m, scm, lmax)
            plm[:,nm-1:(nm-lmax-1):-1,:] = plm[:,1:lmax+1,:]
            
            # Compute spherical harmonic coefficients:
            blm = n.zeros((nm,lmax+1), dtype='complex')
            for l in range(lmax+1):
                blm[:lmax+1,l] = n.sum(plm[:,:lmax+1,l]*brm3[:,:lmax+1]*dsm, axis=0)
                blm[lmax+1:,l] = n.sum(plm[:,lmax+1:,l]*brm3[:,-lmax:]*dsm, axis=0)
                # Apply smoothing filter:
                blm[:,l] *= n.exp(-smooth*l*(l+1))
    
            # Invert transform:
            brm3[:,:] = 0.0
            for j in range(nsm):
                brm3[j,:lmax+1] = n.sum(blm[:lmax+1,:]*plm[j,:lmax+1,:], axis=1)
                brm3[j,-lmax:] = n.sum(blm[lmax+1:,:]*plm[j,lmax+1:,:], axis=1)
    
            brm3 = n.real(n.fft.ifft(brm3, axis=1))
                    
        # (3) INTERPOLATE CENTRAL MAP TO COMPUTATIONAL GRID
        # -------------------------------------------------
        # Form computational grid arrays:
        ds = 2.0/ns
        dph = 2*n.pi/nph
        sc = n.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)  
        pc1 = n.linspace( 0.5*dph, 2*n.pi - 0.5*dph, nph)
        pc = pc1/3 + 2*n.pi/3  # coordinate on the stitched grid
        
        # Interpolate to the computational grid:
        bri = interp2d(pcm, scm, brm3, kind='cubic', copy=True, bounds_error=False, fill_value=0)
        br = n.zeros((ns, nph))
        for i in range(ns):
            br[i,:] = bri(pc, sc[i]).flatten()
        del(brm3, bri)
        
        # (4) CORRECT FLUX BALANCE
        # ------------------------
        br_bound = correct_flux_multiplicative(br)
                    
        return br_bound
    
    
    def ghosts(self,br,bs,bp): #extends the magnetic field as appropriate to give ghost points
        brg = n.zeros((self.nr+1,self.ns+2,self.np+2))
        brg[:,1:-1,1:-1] = br
        brg[:,:,0] = brg[:,:,-2]; brg[:,:,-1] = brg[:,:,1]  #periodic boundary
        brg[:,0,1:self.np//2+1] = brg[:,1,self.np//2+1:-1];brg[:,0,self.np//2+1:-1] = brg[:,1,1:self.np//2+1]  #top/bottom points
        
        bsg = n.zeros((self.nr+2,self.ns+1,self.np+2))
        bsg[1:-1,:,1:-1] = bs
        bsg[:,:,0] = bsg[:,:,-2]; bsg[:,:,-1] = bsg[:,:,1]  #periodic boundary
        bsg[0,:,:] = 2*bsg[1,:,:] - bsg[2,:,:]; bsg[-1,:,:] = 2*bsg[-2,:,:] - bsg[-3,:,:]  #linear over the top/bottom
        
        bpg = n.zeros((self.nr+2,self.ns+2,self.np+1))
        bpg[1:-1,1:-1,:] = bp
        bpg[:,0,:self.np//2] = -bpg[:,1,self.np//2:-1]; bpg[:,0,self.np//2:-1] = -bpg[:,1,:self.np//2] #top/bottom points
        bpg[0,:,:] = 2*bpg[1,:,:] - bpg[2,:,:]; bpg[-1,:,:] = 2*bpg[-2,:,:] - bpg[-3,:,:]    
        return brg,bsg,bpg
    
    def b_to_gridpts(self,brg,bsg,bpg):  #average to gridpoints, for the plotter. Uses ghost point values obtained from ghosts.
        br0 = n.zeros((self.nr+1,self.ns+1,self.np+1))
        br0 = 0.25*(brg[:,1:,1:] + brg[:,:-1,:-1]+brg[:,1:,:-1] + brg[:,:-1,1:])
        bs0 = br0*0;     bp0 = br0*0
        bs0 = 0.25*(bsg[1:,:,1:] + bsg[1:,:,:-1] + bsg[:-1,:,:-1] + bsg[:-1,:,1:])
        bp0 = 0.25*(bpg[1:,1:,:] + bpg[1:,:-1,:] + bpg[:-1,1:,:] + bpg[:-1,:-1,:])
        return br0,bs0,bp0
    
    def b_to_faces(self,br,bs,bp):  #average to faces, to calculate the energy
        br1 = 0.5*(br[1:,:,:] - br[:-1,:,:])
        bs1 = 0.5*(bs[:,1:,:] - bs[:,:-1,:])
        bp1 = 0.5*(bp[:,:,1:] - bp[:,:,:-1])
        print(n.shape(br), n.shape(bs),n.shape(bp))
        return br1,bs1,bp1
        
    def bggrid(self,filename, r, th, ph, brg, bsg, bpg):
        """
            Magnetic field components co-located at grid points.
        """
        
        nr = n.size(r) - 1
        ns = n.size(th) - 1
        np = n.size(ph) - 1   
        
        fid = netcdf.netcdf_file(filename, 'w')
        fid.createDimension('r', nr+1)
        fid.createDimension('th', ns+1)   
        fid.createDimension('ph', np+1)      
        vid = fid.createVariable('r', 'd', ('r',))
        vid[:] = r
        vid = fid.createVariable('th', 'd', ('th',))
        vid[:] = th
        vid = fid.createVariable('ph', 'd', ('ph',))
        vid[:] = ph     
        vid = fid.createVariable('br', 'd', ('ph','th','r'))
        vid[:] = brg
        vid = fid.createVariable('bth', 'd', ('ph','th','r'))
        vid[:] = -bsg   
        vid = fid.createVariable('bph', 'd', ('ph','th','r'))
        vid[:] = bpg    
        fid.close()
        print('Wrote B at grid points to file '+filename)

    def bgstagger(self,filename, br, bs, bp):
        """
            Magnetic field components co-located at staggered grid points (no ghosts)
        """
        r = n.exp(self.rs)
        th = n.arccos(self.ss)
        ph = self.ps
        
        rc = n.exp(self.rc[1:-1])
        thc = n.arccos(self.sc[1:-1])
        phc = self.pc[1:-1]
                
        nr = n.size(r) - 1
        ns = n.size(th) - 1
        np = n.size(ph) - 1   
                
        fid = netcdf.netcdf_file(filename, 'w')
        
        fid.createDimension('r', nr+1)
        fid.createDimension('th', ns+1)   
        fid.createDimension('ph', np+1)      
        
        fid.createDimension('rc', nr)
        fid.createDimension('thc', ns)   
        fid.createDimension('phc', np)      

        vid = fid.createVariable('r', 'd', ('r',))
        vid[:] = r
        vid = fid.createVariable('th', 'd', ('th',))
        vid[:] = th
        vid = fid.createVariable('ph', 'd', ('ph',))
        vid[:] = ph     
        
        vid = fid.createVariable('rc', 'd', ('rc',))
        vid[:] = rc
        vid = fid.createVariable('thc', 'd', ('thc',))
        vid[:] = thc
        vid = fid.createVariable('phc', 'd', ('phc',))
        vid[:] = phc    


        vid = fid.createVariable('br', 'd', ('phc','thc','r'))
        vid[:] = br
        vid = fid.createVariable('bth', 'd', ('phc','th','rc'))
        vid[:] = -bs 
        vid = fid.createVariable('bph', 'd', ('ph','thc','rc'))
        vid[:] = bp  
        fid.close()
        print('Wrote B at staggered grid points to file '+filename)


