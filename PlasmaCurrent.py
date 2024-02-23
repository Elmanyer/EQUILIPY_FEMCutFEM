""" This script contains the routines concerning the Grad-Shafranov right-hand-side source term modelising the 
plasma azimuthal (toroidal) current. """

def Jphi(mu0,R,Z,phi):
    
    jphi = R/mu0
    
    return jphi