""" This script contains the elements describing Gauss quadratures for numerical integration of different orders. 
That is, for different orders of quadrature the integration nodes and integration weights are defined in this script. """

import numpy as np

def GaussQuadrature(element,order):
    """ Obtain Gauss quadrature for reference element and selected quadrature order. 
        Input: - element: type of element -> 0: triangle  ; 1: quadrilateral 
               - order: quadrature order 
        Output: - z: Gauss nodal coordinates matrix 
                - w: Gauss integration weights """
    match element:
            case 2:  # LINE (1D ELEMENT)
                match order:
                    case 1:
                        Ng = 1
                        z = 0
                        w = 2
                    case 2:
                        Ng = 2
                        sq = 1/np.sqrt(3)
                        z = np.array([[-sq], [sq]])
                        w = np.ones([Ng])
                    case 3:
                        Ng = 3
                        sq = np.sqrt(3/5)
                        z = np.array([[-sq], [0], [sq]])
                        w = np.array([5/9, 8/9, 5/9])
            case 0:  # TRIANGLE
                match order:
                    case 1:
                        Ng = 1
                        z = np.array([1/3, 1/3])
                        w = 1/2
                    case 2:   
                        Ng = 3
                        z = np.zeros([Ng,2])
                        z[0,:] = [0.5, 0.5]
                        z[1,:] = [0.5, 0]
                        z[2,:] = [0, 0.5]
                        w = np.ones(Ng)*(1/6)
                    case 3:  
                        Ng = 4  
                        z = np.zeros([Ng,2])
                        z[0,:] = [0.2, 0.2]
                        z[1,:] = [0.6, 0.2]
                        z[2,:] = [0.2, 0.6]
                        z[3,:] = [1/3, 1/3]
                        w = np.array([25/96, 25/96, 25/96, -27/96])
            case 1:  # QUADRILATERAL
                match order:
                    case 3:
                        Ng = 4
                        z = np.zeros([Ng,2])
                        sq = 1/np.sqrt(3)
                        z[0,:] = [-sq, -sq]
                        z[1,:] = [sq, -sq]
                        z[2,:] = [sq, sq]
                        z[2,:] = [-sq, sq]
                        w = np.ones(Ng)
                    case 5:  
                        Ng = 9
                        z = np.array([[0.774596669241483, 0.774596669241483],
                                        [0, 0.774596669241483],
                                        [-0.774596669241483, 0.774596669241483],
                                        [0.774596669241483, 0],
                                        [0, 0],
                                        [-0.774596669241483, 0],
                                        [0.774596669241483, -0.774596669241483],
                                        [0, -0.774596669241483],
                                        [-0.774596669241483, -0.774596669241483]]);
                        w =np.array([0.308641975308641, 0.493827160493826, 0.308641975308641, 0.493827160493826, 0.790123456790123, 0.493827160493826, 0.308641975308641, 0.493827160493826, 0.308641975308641]);
                    

    return z, w, Ng