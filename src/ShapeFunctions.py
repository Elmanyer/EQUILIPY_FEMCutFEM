""" This script contains the information regarding the FEM interpolating shape functions. """

import numpy as np

def ShapeFunctionsReference(X, elemType, elemOrder, node):
    """ Shape functions in reference element, for different type of elements (geometry and order)
    Input: - X: coordinates of point on which to evaluate shape function (natural coordinates) 
           - elemType: 0=line, 1=tri, 2=quad
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
        case 0:    # LINE (1D ELEMENT)
            xi = X
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
    
        case 1:   # TRIANGLE
            xi = X[0]
            eta = X[1]
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

        case 2:    # QUADRILATERAL
            xi = X[0]
            eta = X[1]
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
                            N = xi*(xi+1)*(1-eta**2)/2
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
    
def ShapeFunctionsPhysical(X, Xe, elemType, elemOrder, node):
    """ Shape functions in physical element, for different type of elements (geometry and order)
    Input: - X: coordinates of point on which to evaluate shape function (natural coordinates)
           - Xe: nodal physical coordinates matrix 
           - elemType: 0=line, 1=tri, 2=quad
           - elemOrder: order of element
           - node: local nodal index 
    Output: - N: shape function evaluated at X
            - dNdxi: shape function derivative respect to xi evaluated at X
            - dNdeta: shape function derivative respect to eta evaluated at X
    """

    N = 0

    match elemType:
        case 0:    # LINE (1D ELEMENT)
            x = X
            match elemOrder:
                case 0:
                    # --1--
                    match node:
                        case 1:
                            N = 1  
                case 1:
                    # 1---2
                    match node:
                        case 1:
                            N = LagrangeMultipliers1D(x,Xe,node)
                        case 2:
                            N = LagrangeMultipliers1D(x,Xe,node)    
                case 2:         
                    # 1---2---3
                    N = LagrangeMultipliers1D(x,Xe,node)
    
        case 1:   # TRIANGLE
            x = X[0]
            y = X[1]
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
                    N = 0

        case 2:    # QUADRILATERAL
            x = X[0]
            y = X[1]
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    N = 0
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    N = 0
    return N

def LagrangeMultipliers1D(x0,X,node):
    """ This function computes the 1D Lagrangian Multiplier corresponding to node "node" evaluated at coordinate x0. """
    n = len(X)
    numerator = np.zeros([n-1])
    denominator = np.zeros([n-1])
    index = 0
    for i in range(n):
        if i == node-1:
            pass
        else:
            numerator[index] = X[i]-x0
            denominator[index] = X[i]-X[node-1]
            index += 1
    multiplier = np.prod(numerator)/np.prod(denominator)
    return multiplier

def EvaluateReferenceShapeFunctions(X, elemType, elemOrder, n):
    """ Function that evaluates the shape functions in the reference space for the selected element type and order at points defined by coordinates X
    Input: - X: coordinates of points on which to evaluate shape functions
           - elemType: 0=line, 1=tri, 2=quad
           - elemOrder: order of element
           - n: number of nodes (shape functions) per element
    Output: - N: shape functions evaluated at points with coordinates X
            - dNdxi: shape functions derivatives respect to xi evaluated at points with coordinates X
            - dNdeta: shape functions derivatives respect to eta evaluated at points with coordinates X
    """

    if elemType == 0:
        Ng = len(X)
    else: 
        Ng = len(X[:,0])
        
    N = np.zeros([Ng,n])
    dNdxi = np.zeros([Ng,n])
    dNdeta = np.zeros([Ng,n])
    
    for i in range(n):
        for ig in range(Ng):
            N[ig,i], dNdxi[ig,i], dNdeta[ig,i] = ShapeFunctionsReference(X[ig,:],elemType, elemOrder, i+1)
            
    return N, dNdxi, dNdeta
    

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