""" This script contains the information regarding the FEM interpolating shape functions. """

import numpy as np

def ShapeFunctionsReference(z, elemType, elemOrder, node):
    """ Shape functions in reference element, for different type of elements (geometry and order)
    Input: - z: coordinates of point on which to evaluate shape function (natural coordinates) 
           - elemType: 0=quad, 1=tri, 2=line
           - elemOrder: order of element
           - node: local nodal index 
    Output: - N: shape function evaluated at Gz
            - dNdxi: shape function derivative respect to xi evaluated at z
            - dNdeta: shape function derivative respect to eta evaluated at z
    """

    N = 0
    dNdxi = 0
    dNdeta = 0

    match elemType:
        case 2:    # LINE (1D ELEMENT)
            xi = z
            match elemOrder:
                case 0:
                    # --1--
                    match node:
                        case 1:
                            N = 1
                            dNdxi = 0
                case 1:
                    # 1---2
                    match node:
                        case 1:
                            N = (1-xi)/2
                            dNdxi = -1/2
                        case 2:
                            N = (1+xi)/2
                            dNdxi = 1/2       
                case 2:         
                    # 1---2---3
                    match node:
                        case 1:
                            N = xi*(xi-1)/2
                            dNdxi = xi-0.5
                        case 2:
                            N = (1-xi**2)
                            dNdxi = -2*xi
                        case 3:
                            N = xi*(xi+1)/2
                            dNdxi = xi+0.5
    
        case 0:   # TRIANGLE
            xi = z[0]
            eta = z[1]
            match elemOrder:
                case 1:
                    # 2
                    # |\
                    # | \
                    # 3--1
                    match node:
                        case 1:
                            N = xi
                            dNdxi = 1
                            dNdeta = 0
                        case 2:
                            N = eta
                            dNdxi = 0
                            dNdeta = 1
                        case 3:
                            N = 1-(xi+eta)
                            dNdxi = -1
                            dNdeta = -1
                case 2:
                    # 2
                    # |\
                    # 5 4
                    # |  \
                    # 3-6-1
                    match node:
                        case 1:
                            N = xi*(2*xi-1)
                            dNdxi = 4*xi-1
                            dNdeta = 0
                        case 2:
                            N = eta*(2*eta-1)
                            dNdxi = 0
                            dNdeta = 4*eta-1
                        case 3:
                            N = (1-2*(xi+eta))*(1-(xi+eta))
                            dNdxi = -3+4*(xi+eta)
                            dNdeta = -3+4*(xi+eta)
                        case 4:
                            N = 4*xi*eta
                            dNdxi = 4*eta
                            dNdeta = 4*xi
                        case 5:
                            N = 4*eta*(1-(xi+eta))
                            dNdxi = -4*eta
                            dNdeta = 4*(1-xi-2*eta)
                        case 6:
                            N = 4*xi*(1-(xi+eta))
                            dNdxi = 4*(1-2*xi-eta)
                            dNdeta = -4*xi

        case 1:    # QUADRILATERAL
            xi = z[0]
            eta = z[1]
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    match node:
                        case 1:
                            N = (1-xi)*(1-eta)/4
                            dNdxi = (eta-1)/4
                            dNdeta = (xi-1)/4
                        case 2:
                            N = (1+xi)*(1-eta)/4
                            dNdxi = (1-eta)/4
                            dNdeta = -(1+xi)/4
                        case 3:
                            N = (1+xi)*(1+eta)/4
                            dNdxi = (1+eta)/4
                            dNdeta = (1+xi)/4
                        case 4:
                            N = (1-xi)*(1+eta)/4
                            dNdxi = -(1+eta)/4
                            dNdeta = (1-xi)/4
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    match node: 
                        case 1:
                            N = xi*(xi-1)*eta*(eta-1)/4
                            dNdxi = (xi-1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi-1)*(eta-1/2)/2
                        case 2:
                            N = xi*(xi+1)*eta*(eta-1)/4
                            dNdxi = (xi+1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi+1)*(eta-1/2)/2
                        case 3:
                            N = xi*(xi+1)*eta*(eta+1)/4
                            dNdxi = (xi+1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi+1)*(eta+1/2)/2
                        case 4:
                            N = xi*(xi-1)*eta*(eta+1)/4
                            dNdxi = (xi-1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi-1)*(eta+1/2)/2
                        case 5:
                            N = (1-xi**2)*eta*(eta-1)/2
                            dNdxi = -xi*eta*(eta-1)
                            dNdeta = (1-xi**2)*(eta-1/2)
                        case 6:
                            N = xi*(xi+1)*(1-eta^2)/2
                            dNdxi = (xi+1/2)*(1-eta**2)
                            dNdeta = xi*(xi+1)*(-eta)
                        case 7:
                            N = (1-xi**2)*eta*(eta+1)/2
                            dNdxi = -xi*eta*(eta+1)
                            dNdeta = (1-xi**2)*(eta+1/2)
                        case 8:
                            N = xi*(xi-1)*(1-eta**2)/2
                            dNdxi = (xi-1/2)*(1-eta**2)
                            dNdeta = xi*(xi-1)*(-eta)
                        case 9:
                            N = (1-xi**2)*(1-eta**2)
                            dNdxi = -2*xi*(1-eta**2)
                            dNdeta = (1-xi**2)*(-2*eta)
    return N, dNdxi, dNdeta


def EvaluateShapeFunctions(elemType, elemOrder, n, z):
    """ 
    Input: - elemType: 0=quad, 1=tri, 2=line
           - elemOrder: order of element
           - n: number of nodes (shape functions) per element
           - z: coordinates of Gauss integration nodes 
    Output: - N: shape functions evaluated at Gauss integration nodes
            - dNdxi: shape functions derivatives respect to xi evaluated at Gauss integration nodes
            - dNdeta: shape functions derivatives respect to eta evaluated at Gauss integration nodes
    """

    if elemType == 2:
        Ng = len(z)
    else: 
        Ng = len(z[:,0])
        
    N = np.zeros([Ng,n])
    dNdxi = np.zeros([Ng,n])
    dNdeta = np.zeros([Ng,n])
    
    for i in range(n):
        for ig in range(Ng):
            N[ig,i], dNdxi[ig,i], dNdeta[ig,i] = ShapeFunctionsReference(z[ig,:],elemType, elemOrder, i+1)
            
    return N, dNdxi, dNdeta
    
    
def ShapeFunctionsPhysical(z, Xe, elemType, elemOrder, node):
    """ Shape functions in physical element, for different type of elements (geometry and order)
    Input: - z: coordinates of point on which to evaluate shape function (natural coordinates)
           - Xe: nodal physical coordinates matrix 
           - elemType: 0=quad, 1=tri, 2=line
           - elemOrder: order of element
           - node: local nodal index 
    Output: - N: shape function evaluated at z
            - dNdxi: shape function derivative respect to xi evaluated at z
            - dNdeta: shape function derivative respect to eta evaluated at z
    """

    N = 0

    match elemType:
        case 2:    # LINE (1D ELEMENT)
            x = z
            match elemOrder:
                case 0:
                    # --1--
                    match node:
                        case 1:
                            N = 1
                            dNdxi = 0
                case 1:
                    # 1---2
                    match node:
                        case 1:
                            N = (1-xi)/2
                            dNdxi = -1/2
                        case 2:
                            N = (1+xi)/2
                            dNdxi = 1/2       
                case 2:         
                    # 1---2---3
                    match node:
                        case 1:
                            N = xi*(xi-1)/2
                            dNdxi = xi-0.5
                        case 2:
                            N = (1-xi**2)
                            dNdxi = -2*xi
                        case 3:
                            N = xi*(xi+1)/2
                            dNdxi = xi+0.5
    
        case 0:   # TRIANGLE
            x = z[0]
            y = z[1]
            match elemOrder:
                case 1:
                    # 2
                    # |\
                    # | \
                    # 3--1
                    v = np.ones([3,1])
                    J = np.concatenate((v, Xe), axis=1)
                    detJ = np.linalg.det(J)
                    match node:
                        case 1:
                            j = 1
                            k = 2
                        case 2:
                            j = 2
                            k = 0
                        case 3:
                            j = 0
                            k = 1
                    N = (Xe[j,0]*Xe[k,1]-Xe[k,0]*Xe[j,1]+(Xe[j,1]-Xe[k,1])*x+(Xe[k,0]-Xe[j,0])*y)/detJ
                case 2:
                    # 2
                    # |\
                    # 5 4
                    # |  \
                    # 3-6-1
                    match node:
                        case 1:
                            N = xi*(2*xi-1)
                            dNdxi = 4*xi-1
                            dNdeta = 0
                        case 2:
                            N = eta*(2*eta-1)
                            dNdxi = 0
                            dNdeta = 4*eta-1
                        case 3:
                            N = (1-2*(xi+eta))*(1-(xi+eta))
                            dNdxi = -3+4*(xi+eta)
                            dNdeta = -3+4*(xi+eta)
                        case 4:
                            N = 4*xi*eta
                            dNdxi = 4*eta
                            dNdeta = 4*xi
                        case 5:
                            N = 4*eta*(1-(xi+eta))
                            dNdxi = -4*eta
                            dNdeta = 4*(1-xi-2*eta)
                        case 6:
                            N = 4*xi*(1-(xi+eta))
                            dNdxi = 4*(1-2*xi-eta)
                            dNdeta = -4*xi

        case 1:    # QUADRILATERAL
            xi = z[0]
            eta = z[1]
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    match node:
                        case 1:
                            N = (1-xi)*(1-eta)/4
                            dNdxi = (eta-1)/4
                            dNdeta = (xi-1)/4
                        case 2:
                            N = (1+xi)*(1-eta)/4
                            dNdxi = (1-eta)/4
                            dNdeta = -(1+xi)/4
                        case 3:
                            N = (1+xi)*(1+eta)/4
                            dNdxi = (1+eta)/4
                            dNdeta = (1+xi)/4
                        case 4:
                            N = (1-xi)*(1+eta)/4
                            dNdxi = -(1+eta)/4
                            dNdeta = (1-xi)/4
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    match node: 
                        case 1:
                            N = xi*(xi-1)*eta*(eta-1)/4
                            dNdxi = (xi-1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi-1)*(eta-1/2)/2
                        case 2:
                            N = xi*(xi+1)*eta*(eta-1)/4
                            dNdxi = (xi+1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi+1)*(eta-1/2)/2
                        case 3:
                            N = xi*(xi+1)*eta*(eta+1)/4
                            dNdxi = (xi+1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi+1)*(eta+1/2)/2
                        case 4:
                            N = xi*(xi-1)*eta*(eta+1)/4
                            dNdxi = (xi-1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi-1)*(eta+1/2)/2
                        case 5:
                            N = (1-xi**2)*eta*(eta-1)/2
                            dNdxi = -xi*eta*(eta-1)
                            dNdeta = (1-xi**2)*(eta-1/2)
                        case 6:
                            N = xi*(xi+1)*(1-eta^2)/2
                            dNdxi = (xi+1/2)*(1-eta**2)
                            dNdeta = xi*(xi+1)*(-eta)
                        case 7:
                            N = (1-xi**2)*eta*(eta+1)/2
                            dNdxi = -xi*eta*(eta+1)
                            dNdeta = (1-xi**2)*(eta+1/2)
                        case 8:
                            N = xi*(xi-1)*(1-eta**2)/2
                            dNdxi = (xi-1/2)*(1-eta**2)
                            dNdeta = xi*(xi-1)*(-eta)
                        case 9:
                            N = (1-xi**2)*(1-eta**2)
                            dNdxi = -2*xi*(1-eta**2)
                            dNdeta = (1-xi**2)*(-2*eta)
    return N


def Jacobian(x,y,dNdxi,dNdeta):
    """ Function that computes the Jacobian of the mapping between physical and natural coordinates 
        Input: - x: nodal x physical coordinates 
               - y: nodal y physical coordinates 
               - dNdxi: shape functions derivatives respect to xi evaluated at Gauss integration nodes
               - dNdeta: shape functions derivatives respect to eta evaluated at Gauss integration nodes
        Output: - invJ: Jacobian inverse
                - detJ: Jacobian determinant 
            """
    n = len(x)
    J = np.zeros([2,2])
    # COMPUTE JACOBIAN
    for i in range(n):
        J += np.array([[x[i]*dNdxi[i], y[i]*dNdxi[i]] ,[x[i]*dNdeta[i], y[i]*dNdeta[i]]])
        # COMPUTE INVERSE MATRIX AND JACOBIAN
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)
    return invJ, detJ

def Jacobian1D(x,y,dNdxi):
    n = len(x)
    J = np.zeros([2])
    for i in range(n):
        J += dNdxi[i]*np.array([x[i],y[i]])
    detJ = np.linalg.norm(J)
    return detJ