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
                    # 1---3---2
                    match node:
                        case 1:
                            N = -xi*(1-xi)/2
                            dNdxi = xi-0.5
                        case 2:
                            N = xi*(xi+1)/2
                            dNdxi = xi+0.5
                        case 3:
                            N = 1-xi**2
                            dNdxi = -2*xi
                case 3:         
                    # 1-3-4-2
                    match node:
                        case 1:
                            N = -9/16*(xi+1/3)*(xi-1/3)*(xi-1)
                            dNdxi = -9/16*((xi-1/3)*(xi-1)+(xi+1/3)*(xi-1)+(xi+1/3)*(xi-1/3))
                        case 2:
                            N =  9/16*(xi+1)*(xi+1/3)*(xi-1/3)
                            dNdxi = 9/16*((xi+1/3)*(xi-1/3)+(xi+1)*(xi-1/3)+(xi+1)*(xi+1/3))
                        case 3:
                            N = 27/16*(xi+1)*(xi-1/3)*(xi-1)
                            dNdxi = 27/16*((xi-1/3)*(xi-1)+ (xi+1)*(xi-1)+ (xi+1)*(xi-1/3))
                        case 4:
                            N = -27/16*(xi+1)*(xi+1/3)*(xi-1)
                            dNdxi = -27/16*((xi+1/3)*(xi-1)+(xi+1)*(xi-1)+(xi+1)*(xi+1/3))
    
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
                case 3:
                    #  2
                    # | \
                    # 6  5 
                    # |   \
                    # 7 10 4
                    # |     \
                    # 3-8--9-1
                    match node:
                        case 1:
                            N = (9/2)*(1/3-xi)*(2/3-xi)*xi
                            dNdxi = -(9/2)*((2/3-xi)*xi+(1/3-xi)*xi-(1/3-xi)*(2/3-xi))
                            dNdeta = 0
                        case 2:
                            N = (9/2)*(1/3-eta)*(2/3-eta)*eta 
                            dNdxi = 0
                            dNdeta = -(9/2)*((2/3-eta)*eta+(1/3-eta)*eta-(1/3-eta)*(2/3-eta))
                        case 3:
                            N = (9/2)*(1-xi-eta)*(2/3-xi-eta)*(1/3-xi-eta)  
                            dNdxi = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                            dNdeta = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                        case 4:
                            N = -3*(9/2)*(1/3-xi)*xi*eta
                            dNdxi = -3*(9/2)*((1/3-xi)*eta-xi*eta)
                            dNdeta = -3*(9/2)*((1/3-xi)*xi)
                        case 5:
                            N = -3*(9/2)*xi*(1/3-eta)*eta 
                            dNdxi = -3*(9/2)*((1/3-eta)*eta)
                            dNdeta = -3*(9/2)*((1/3-eta)*xi-xi*eta)
                        case 6:
                            N = -3*(9/2)*(1-xi-eta)*(1/3-eta)*eta
                            dNdxi = 3*(9/2)*((1/3-eta)*eta)
                            dNdeta = -3*(9/2)*(-(1/3-eta)*eta-(1-xi-eta)*eta+(1-xi-eta)*(1/3-eta))
                        case 7:
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*eta
                            dNdxi = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta)
                            dNdeta = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta+(1-xi-eta)*(2/3-xi-eta))
                        case 8:
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*xi 
                            dNdxi = 3*(9/2)*((1-xi-eta)*(2/3-xi-eta)-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                            dNdeta = 3*(9/2)*(-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                        case 9:
                            N = -3*(9/2)*(1-xi-eta)*(1/3-xi)*xi 
                            dNdxi = -3*(9/2)*((1-xi-eta)*(1/3-xi)-(1-xi-eta)*xi-(1/3-xi)*xi)
                            dNdeta = -3*(9/2)*(-(1/3-xi)*xi)
                        case 10:
                            N = 6*(9/2)*(1-xi-eta)*xi*eta
                            dNdxi = 6*(9/2)*((1-xi-eta)*eta-xi*eta)
                            dNdeta = 6*(9/2)*((1-xi-eta)*xi-xi*eta)
                            
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
        J += np.array([[x[i]*dNdxi[i], y[i]*dNdxi[i]] ,
                       [x[i]*dNdeta[i], y[i]*dNdeta[i]]])
        # COMPUTE INVERSE MATRIX AND JACOBIAN
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)
    return J, invJ, detJ


def Jacobian1D(x,y,dNdxi):
    n = len(x)
    J = np.zeros([2])
    for i in range(n):
        J += dNdxi[i]*np.array([x[i],y[i]])
    detJ = np.linalg.norm(J)
    return detJ