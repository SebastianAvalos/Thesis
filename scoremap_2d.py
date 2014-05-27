#S. Avalos Filter library

import numpy as np

class Scoremap(object):
    def __init__(self,ngridx,ngridy,nmgrid):
         
        
        self.avtot = -99.0*np.ones((ngridx,ngridy,nmgrid))
        self.avNS = -99.0*np.ones((ngridx,ngridy,nmgrid))
        self.avEW = -99.0*np.ones((ngridx,ngridy,nmgrid))
        #self.avELE = -99.0*np.ones((ngridx,ngridy,ngridz,nmgrid))                      
        self.curNS = -99.0*np.ones((ngridx,ngridy,nmgrid)) 
        self.curEW = -99.0*np.ones((ngridx,ngridy,nmgrid))
        #self.curELE = -99.0*np.ones((ngridx,ngridy,ngridz,nmgrid))
        self.gradNS = -99.0*np.ones((ngridx,ngridy,nmgrid)) 
        self.gradEW = -99.0*np.ones((ngridx,ngridy,nmgrid))
        #self.gradELE = -99.0*np.ones((ngridx,ngridy,ngridz,nmgrid))  
