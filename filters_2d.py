#S. Avalos Filter library

import numpy as np

class Filters(object):
    def __init__(self,ngridx,ngridy):
        if ngridx == 1:
            nx=1
        else:
            nx=(ngridx-1)/2 
        
        if ngridy == 1:
            ny=1
        else:
            ny=(ngridy-1)/2
        
        
        self.avtot = np.zeros((ngridx,ngridy))
        self.avNS = np.zeros((ngridx,ngridy))
        self.avEW = np.zeros((ngridx,ngridy))
        #self.avELE = np.zeros((ngridx,ngridy))                      
        self.curNS = np.zeros((ngridx,ngridy)) 
        self.curEW = np.zeros((ngridx,ngridy))
        #self.curELE = np.zeros((ngridx,ngridy))
        self.gradNS = np.zeros((ngridx,ngridy)) 
        self.gradEW = np.zeros((ngridx,ngridy))
        #self.gradELE = np.zeros((ngridx,ngridy))       
        
        for x in range(-nx,nx+1):
            for y in range(-ny,ny+1):
                #for z in range(-nz,nz+1):
                self.avNS[(x+nx),(y+ny)] = (1.0-(1.0*np.absolute(y)/(ny)))
                self.avEW[(x+nx),(y+ny)] = (1.0-(1.0*np.absolute(x)/(nx)))
                #self.avELE[(x+nx),(y+ny),(z+nz)] = (1.0-(1.0*np.absolute(z)/(nz)))
                self.curNS[(x+nx),(y+ny)] = (1.0*y/ny)
                self.curEW[(x+nx),(y+ny)] = (1.0*x/nx)
                #self.curELE[(x+nx),(y+ny),(z+nz)] = (1.0*z/nz)
                self.gradNS[(x+nx),(y+ny)] = (2.0*np.absolute(y)/(ny)-1.0)
                self.gradEW[(x+nx),(y+ny)] = (2.0*np.absolute(x)/(nx)-1.0)
                #self.gradELE[(x+nx),(y+ny),(z+nz)] = (2.0*np.absolute(z)/(nz)-1.0)
                self.avtot[(x+nx),(y+ny)] = (1.0/(ngridx*ngridy))
    
