""" This script contains the Python object defining a plasma equilibrium problem, modeled using the Grad-Shafranov equation
in an axisymmetrical system such as a tokamak. """

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from GaussQuadrature import *
from ShapeFunctions import *
from PlasmaCurrent import *

class Equili:
    
    # GENERAL PARAMETERS
    epsilon0 = 8.8542E-12       # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19               # J eV-1   Botlzmann constant

    def __init__(self,folder_loc,ElementType,ElementOrder,Rmax,Rmin,epsilon,kappa,delta):
        self.directory = folder_loc
        self.case = folder_loc[folder_loc.rfind("/")+1:]
        
        # DECLARE ATTRIBUTES
        self.ElType = ElementType
        self.ElOrder = ElementOrder
        self.epsilon = epsilon
        self.kappa = kappa
        self.delta = delta
        self.Rmax = Rmax
        self.Rmin = Rmin
        self.R0 = (Rmax+Rmin)/2
        self.TOL_inner = 1e-3
        self.TOL_outer = 1e-3
        self.itmax = 5
        self.beta = 1e3  # NITSCHE'S METHOD PENALTY TERM

        return
    
    def ReadMesh(self):
        # NUMBER OF NODES PER ELEMENT
        match self.ElType:
            case 0:
                match self.ElOrder:
                    case 1:
                        self.n = 3
                    case 2: 
                        self.n = 6
                    case 3:
                        self.n = 10
            case 1:
                match self.ElOrder:
                    case 1:
                        self.n = 4
                    case 2:
                        self.n = 9
                    case 3:
                        self.n = 16
        
        # READ DOM FILE .dom.dat
        MeshDataFile = self.directory +'/'+ self.case +'.dom.dat'
        self.Nn = 0   # number of nodes
        self.Ne = 0   # number of elements
        file = open(MeshDataFile, 'r') 
        for line in file:
            l = line.split('=')
            if l[0] == '  NODAL_POINTS':  # read number of nodes
                self.Nn = int(l[1])
            elif l[0] == '  ELEMENTS':  # read number of elements
                self.Ne = int(l[1])
            elif l[0] == '  SPACE_DIMENSIONS':  # read space dimensions 
                self.dim = int(l[1])
            elif l[0] == '  BOUNDARIES':  # read number of boundaries
                self.Nbound = int(l[1])
        file.close()
        
        # READ MESH FILE .geo.dat
        MeshFile = self.directory +'/'+ self.case +'.geo.dat'
        self.T = np.zeros([self.Ne,self.n], dtype = int)
        self.X = np.zeros([self.Nn,self.dim], dtype = float)
        file = open(MeshFile, 'r') 
        i = -1
        j = -1
        for line in file:
            # first we format the line read in order to remove all the '\n'  
            l = line.split(' ')
            l = [k for k in l if k != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            # We identify when the connectivity information starts
            if l[0] == 'ELEMENTS':
                i=0
                continue
            # We identify when the connectivity information ends
            elif l[0] == 'END_ELEMENTS':
                i=-1
                continue
            # We identify when the coordinates information starts
            elif l[0] == 'COORDINATES':
                j=0
                continue
            # We identify when the coordinates information ends
            elif l[0] == 'END_COORDINATES':
                j=-1
                continue
            if i>=0:
                for k in range(self.n):
                    self.T[i,k] = int(l[k+1])
                i += 1
            if j>=0:
                for k in range(self.dim):
                    self.X[j,k] = float(l[k+1])
                j += 1
        file.close()
        # PYTHON INDEXES START AT 0 AND NOT AT 1. THUS, THE CONNECTIVITY MATRIX INDEXES MUST BE MODIFIED
        self.T = self.T -1
        
        return
    
    def ComputeLinearSolutionCoefficients(self):
        """ Computes the coeffients for the magnetic flux in the linear source term case, that is for 
                    GRAD-SHAFRANOV EQ:  DELTA*(PHI) = R^2
            for which the exact solution is 
                    PHI = R^4/8 + D1 + D2*R^2 + D3*(R^4-4*R^2*Z^2)
                This function returns coefficients D1, D2, D3
                    
            Input: - epsilon: magnetic confinement cross-section inverse aspect ratio
                   - kappa: magnetic confinement cross-section elongation
                   - delta: magnetic confinement cross-section triangularity """
                
        A = np.array([[1, (1+self.epsilon)**2, (1+self.epsilon)**4], 
                    [1, (1-self.epsilon)**2, (1-self.epsilon)**4],
                    [1, (1-self.delta*self.epsilon)**2, (1-self.delta*self.epsilon)**4-4*(1-self.delta*self.epsilon)**2*self.kappa**2*self.epsilon**2]])
        b = -(1/8)*np.array([[(1+self.epsilon)**4], [(1-self.epsilon)**4], [(1-self.delta*self.epsilon)**4]])
        
        coeffs = np.linalg.solve(A,b)
        return coeffs 
    
    def InitialGuess(self):
        """ Use the analytical solution for the LINEAR case as initial guess. The plasma region is characterised by a negative phi in this solution. """
        # ADIMENSIONALISE MESH
        Xstar = self.X/self.R0
        phi0 = np.zeros([self.Nn])
        self.coeffs = self.ComputeLinearSolutionCoefficients()
        for i in range(self.Nn):
            phi0[i] = Xstar[i,0]**4/8 + self.coeffs[0] + self.coeffs[1]*Xstar[i,0]**2 + self.coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
        return phi0
    
    
    def ClassifyElements(self):
        """ Function that sperates the elements into 3 groups: 
                - PlasmaElems: elements inside the plasma region P(phi) where the plasma current is different from 0
                - VacuumElems: elements outside the plasma region P(phi) where the plasma current is 0
                - InterElems: elements containing the plasma region's interface """
        
        self.PlasmaElems = np.zeros([self.Ne], dtype=int)
        self.VacuumElems = np.zeros([self.Ne], dtype=int)
        self.InterElems = np.zeros([self.Ne], dtype=int)
        kplasm = 0
        kvacuu = 0
        kint = 0
                
        for e in range(self.Ne):
            LSe = self.LevelSet[self.T[e,:]]  # elemental nodal level-set values
            for i in range(self.n-1):
                if np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between nodal values change -> interface element
                    self.InterElems[kint] = e
                    kint += 1
                    break
                else:
                    if i+2 == self.n:   # if all nodal values hasve the same sign
                        if np.sign(LSe[i+1]) > 0:   # all nodal values with positive sign -> vacuum vessel element
                            self.VacuumElems[kvacuu] = e
                            kvacuu += 1
                        else:   # all nodal values with negative sign -> plasma region element 
                            self.PlasmaElems[kplasm] = e
                            kplasm += 1
                            
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.InterElems = self.InterElems[:kint]
        return
    
    
    def InterfaceLinearApproximation(self, element_index):
        """ Function computing the intersection points between the element edges and the interface (for elements containing the interface) 
            FOR THE MOMENT, DESIGNED EXCLUSIVELY FOR TRIANGULAR ELEMENTS
        Input: - element_index: index of element for which to compute approximated interface coordinates 
        Output: - interface: matrix containing the coordinates of points located at the intrsection between the interface and the element's edges
                - Te: modified elemental conectivity matrix, where the first entry corresponds to the node "alone" in its respective region (common node 
                    to both edges intersecting with interface), and the following entries correspond to the other elemental nodes, which together with the
                    first one, define the edges intersecting the interface. """
        # READ CONNECTIVITIES
        Te = self.T[element_index,:]
        # READ NODAL COORDINATES 
        Xe = self.X[Te,:]
        # READ LEVEL-SET NODAL VALUES
        LSe = self.LevelSet[Te]  
        # LOOK FOR THE NODE WHICH HAS DIFFERENT SIGN...
        pospos = np.where(LSe > 0)[0]
        posneg = np.where(LSe < 0)[0]
        # ... PIVOT COORDINATES MATRIX ACCORDINGLY
        if len(pospos) > len(posneg):  # 2 nodal level-set values are positive (outside plasma region)
            pos = np.concatenate((posneg,pospos),axis=0)
            Te = Te[pos]
            Xe = Xe[pos]
            LSe = LSe[pos]
        else: # 2 nodal level-set values are negative (inside plasma region)
            pos = np.concatenate((pospos,posneg),axis=0)
            Te = Te[pos]
            Xe = Xe[pos]
            LSe = LSe[pos]

        # NOW, THE FIRST ROW IN Xe AND FIRST ELEMENT IN LSe CORRESPONDS TO THE NODE ALONE IN ITS RESPECTIVE REGION (INSIDE OR OUTSIDE PLASMA REGION)

        # WE DEFINE NOW THE DIFFERENT FUNCTION WE NEED IN ORDER TO BUILD THE TRANSCENDENTAL EQUATION CHARACTERISING THE INTERSECTION BETWEEN
        # THE ELEMENT'S EDGE AND THE LEVEL-SET 0-CONTOUR
        def z(r,Xe,edge):
            # FUNCTION DESCRIBING THE RESTRICCION ASSOCIATED TO THE ELEMENT EDGE
            z = ((Xe[edge,1]-Xe[0,1])*r+Xe[0,1]*Xe[edge,0]-Xe[edge,1]*Xe[0,0])/(Xe[edge,0]-Xe[0,0])
            return z

        def fun(r,Xe,LSe,edge):
            def N0(r,z,Xe):
                # SHAPE FUNCTION IN PHYSICAL SPACE FOR NODE WHICH IS "ALONE" IN RESPECTIVE REGION (OUTSIDE OR INSIDE PLASMA REGION)
                j = 1
                k = 2
                N = Xe[j,0]*Xe[k,1]-Xe[k,0]*Xe[j,1]+(Xe[j,1]-Xe[k,1])*r+(Xe[k,0]-Xe[j,0])*z
                return N
            def Nedge(r,z,Xe,edge):
                # SHAPE FUNCTION IN PHYSICAL SPACE FOR NODE ALONG THE EDGE FOR WHICH FIND THE INTERSECTION WITH LEVEL-SET 0-CONTOUR
                j = (edge+1)%3
                k = (edge+2)%3
                N = Xe[j,0]*Xe[k,1]-Xe[k,0]*Xe[j,1]+(Xe[j,1]-Xe[k,1])*r+(Xe[k,0]-Xe[j,0])*z
                return N
            
            # TRANSCENDENTAL EQUATION TO SOLVE
            f = N0(r,z(r,Xe,edge),Xe)*LSe[0] + Nedge(r,z(r,Xe,edge),Xe,edge)*LSe[edge]
            return f

        # SOLVE TRANSCENDENTAL EQUATION FOR BOTH EDGES AND OBTAIN INTERSECTION COORDINATES
        interface = np.zeros([2,2])
        for i, edge in enumerate([1,2]):
            sol = optimize.root(fun, Xe[0,0], args=(Xe,LSe,edge))
            interface[i,:] = [sol.x, z(sol.x,Xe,edge)]
        
        return interface, Te
    
    def ComputeInterfaceCoordinates(self):
        """ Compute the coordinates for the points describing the interface linear approximation 
        The coordinates are organised in a 3D matrix as follows:
                InterfaceCoordinates = [[[x11 y11]   # coordinates point 1 interface 1
                                         [x12 y12]]  # coordinates point 2 interface 1
                                        
                                        [[x21 y21]   # coordinates point 1 interface 2
                                         [x22 y22]]  # coordinates point 2 interface 2
                                        
                                        [[x31 y31]   # coordinates point 1 interface 3
                                          x31 y31]]  # coordinates point 2 interface 3 
                                            ...     
                                                                
        On the other hand, this function also returns the mofidied elemental connectivities, such that 
                Tinter = [[NELinterface1 NNODEcommonvertex1 NNODEedge11 NNODEedge12]
                          [NELinterface2 NNODEcommonvertex2 NNODEedge21 NNODEedge22]
                          [NELinterface3 NNODEcommonvertex3 NNODEedge31 NNODEedge32]
                                                  ... 
                    where: -> NELinterface-i is the element global index for the i-th interface 
                           -> NNODEcommonvertex-i is the global index for the node which is common to both edges intersecting with the i-th interface
                           -> NNODEedge-ij is the global index for the node defining the j-th edge intersecting the i-th interface """

        Nintersections = 2 # 2D can only be 2 intersections of interface with element edges
        self.InterfaceCoordinates = np.zeros([len(self.InterElems),Nintersections,self.dim])
        self.Tinter = np.zeros([len(self.InterElems), self.n+1], dtype= int)

        for i, element in enumerate(self.InterElems):
            interface, Te = self.InterfaceLinearApproximation(element)
            self.InterfaceCoordinates[i,:,:] = interface
            self.Tinter[i,0] = element
            self.Tinter[i,1:] = Te 

        return
    
    def InverseMapping(self, X, element):
        """ This function represents the inverse mapping corresponding to the transformation from natural to physical coordinates. 
        That is, given a point in physical space with coordinates X in the specified element, this function returns the point mapped
        in the natural reference frame with coordinates Xg. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric mapping equations. 
        
        Input: - X: physical nodal coordinates 
               - element: index of element 
        Output: - Xg: mapped reference nodal coodinates """
        
        # READ ELEMENTAL NODAL COORDINATES 
        Xe = self.X[self.T[element,:],:]
        # DEFINE THE NONLINEAR SYSTEM 
        def fun(Xg, X, Xe):
            f = np.array([-X[0],-X[1]])
            for i in range(self.n):
                Nig, foo, foo = ShapeFunctionsReference(Xg,self.ElType, self.ElOrder, i+1)
                f[0] += Nig*Xe[i,0]
                f[1] += Nig*Xe[i,1]
            return f
        # SOLVE USING NONLINEAR SOLVER
        Xg0 = np.array([1/2, 1/2])  # INITIAL GUESS FOR ROOT SOLVER
        sol = optimize.root(fun, Xg0, args=(X,Xe))
        Xg = sol.x
        return Xg
    
    
    def SubdivideTriangularInterfaceElement(self, interface):
        """ Function computing the subdivision elements 
        Input: - interface: index of the interface contained in the element 
        Output: - Xemod: nodal coordinates matrix for element containing interface, such that the last 2 rows correspond to the intersection points between interface and element edges
                - Temod: local connectivity matrix for 3 triangular subelements 
                - Dom: array indicating in which region (unside or outside) is located each subtriangle
                - PHIemod: nodal PHI values (inner iteration n) for nodes in cut element, such that the last 2 elements correspond to the intersection points between interface and element edges"""
        
        Xe = self.X[self.Tinter[interface,:],1:]   # nodal coordinates of element, where the first row is the node common to both edges intersecting with the interface
        Xeint = self.InterfaceCoordinates[interface,:,:]   # coordinates of intersections between interface and edges

        # MODIFIED NODAL MATRIX AND CONECTIVITIES, ACCOUNTING FOR 3 SUBTRIANGLES 
        Xemod = np.concatenate((Xe, Xeint), axis=0)
        Temod = np.zeros([3, self.n], dtype = int)  # local connectivities for 3 subtriangles
        PHIemod = np.zeros([self.n+2])

        Temod[0,:] = [0, 3, 4]  # first triangular subdomain is common node and intersection nodes
        if self.phi_inner0[self.Tinter[interface,1]] < 0:  # COMMON NODE YIELD PHI < 0 -> INSIDE REGION
            Dom = np.array([-1,1,1])
        else:
            Dom = np.array([1,-1,-1])
            
        # COMPUTE PHI VALUES AT THE NODES IN ELEMENT AND INTERSECTION BETWEEN INTERFACE AND EDGES
        PHIemod[:3] = self.phi_inner0[self.Tinter[interface,1:]]  # the first 3 values correspond to the nodal phi values 
        # INTERPOLATE TO FIND VALUES AT INTERFACE INTERSECTIONS
        for i in range(self.n):
            PHIemod[3] += ShapeFunctionsPhysical(Xeint[0,:], Xe, self.elemType, self.elemOrder, i+1)*self.phi_inner0[self.Tinter[interface,i+1]]  # first intersection
            PHIemod[4] += ShapeFunctionsPhysical(Xeint[1,:], Xe, self.elemType, self.elemOrder, i+1)*self.phi_inner0[self.Tinter[interface,i+1]]  # second interection

        # PREPARE TESSELATION
        # COMPARE DISTANCE INTERFACE-(EDGE NODE)
        edge = 1
        distance1 = np.linalg.norm(Xeint[edge-1,:]-Xe[edge,:])
        edge = 2
        distance2 = np.linalg.norm(Xeint[edge-1,:]-Xe[edge,:])

        if distance1 < distance2:
            Temod[1,:] = [3, 1, 2]
            Temod[2,:] = [3, 4, 2]
        if distance1 > distance2:
            Temod[1,:] = [4, 2, 1]
            Temod[2,:] = [4, 3, 1]
            
        return Xemod, Temod, Dom, PHIemod
    
    
    def CheckNodeOnEdge(self,x,Xe,TOL):
        """ Function which checks if point with coordinates x is on any edge of the element with nodal coordinates Xe. """
        n = np.shape(Xe)[0]
        edgecheck = False
        for edge in range(n):
            i = edge
            j = (edge+1)%n
            if abs(Xe[j,0]-Xe[i,0]) < 1e-6:  # infinite slope <=> vertical edge
                if abs(Xe[i,0]-x[0]) < TOL:
                    edgecheck = True
                    break
            y = lambda x : ((Xe[j,1]-Xe[i,1])*x+Xe[i,1]*Xe[j,0]-Xe[j,1]*Xe[i,0])/(Xe[j,0]-Xe[i,0])  # function representing the restriction on the edge
            if abs(y(x[0])-x[1]) < TOL:
                edgecheck = True
                break
        if edgecheck == True:
            return i, j
        else:
            return "Point not on edges"
        
    
    def Tessellation(self,Xe,Xeint):
        """ This function performs the TESSELLATION of an element with nodal coordinates Xe and interface coordinates Xeint (intersection with edges) 
        Input: - Xe: element nodal coordinates 
               - Xeint: coordinates of intersection points between interface and edges 
        Output: - XeTESS: Nodal coordinates matrix storing the coordinates of the element vertex and interface points 
                - TeTESS: Tessellation connectivity matrix such that 
                        TeTESS = [[Connectivities for subelement 1]
                                  [Connectivities for subelement 2]
                                                ...                            
                """
                
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        Nint = np.shape(Xeint)[0]  # number of intersection points
        edgenodes = np.zeros(np.shape(Xeint), dtype=int)
        nodeedgeinter = np.zeros([Nint], dtype=int)
        for i in range(Nint):
            edgenodes[i,:] = self.CheckNodeOnEdge(Xeint[i,:],Xe,1e-4)
        commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop()
        for i in range(Nint):
            edgenodesset = set(edgenodes[i,:])
            edgenodesset.remove(commonnode)
            nodeedgeinter[i] = edgenodesset.pop()
        
        Xe = Xe[np.concatenate((np.array([commonnode]), nodeedgeinter), axis=0),:]

        # MODIFIED NODAL MATRIX AND CONECTIVITIES, ACCOUNTING FOR 3 SUBTRIANGLES 
        XeTESS = np.concatenate((Xe, Xeint), axis=0)
        TeTESS = np.zeros([3, self.n], dtype = int)  # connectivities for 3 subtriangles

        TeTESS[0,:] = [0, 3, 4]  # first triangular subdomain is common node and intersection nodes

        # COMPARE DISTANCE INTERFACE-(EDGE NODE)
        edge = 1
        distance1 = np.linalg.norm(Xeint[edge-1,:]-Xe[edge,:])
        edge = 2
        distance2 = np.linalg.norm(Xeint[edge-1,:]-Xe[edge,:])

        if distance1 <= distance2:
            TeTESS[1,:] = [3, 1, 2]
            TeTESS[2,:] = [3, 4, 2]
        if distance1 > distance2:
            TeTESS[1,:] = [4, 2, 1]
            TeTESS[2,:] = [4, 3, 1]
        
        return XeTESS, TeTESS
    
    
    def ComputeModifiedQuadrature2D(self,interface):
        """ This function returns the modified 2D Gauss quadrature to integrate over an interface element. 
        Input: - interface: index of the interface for which compute modified quadrature
        Output: - Xemod: nodal coordinates matrix for tessellated physical element
                - Temod: connectivity matrix for subelements in tessellated physical element
                - XgmodREF: Gaus integration nodal coordinates for subelements in tessellated reference element
                - Dom: array specifying the region, inside or outside, on which each physical subelement falls
        
        Important quantities in this routine:
        ### 2D REFERENCE ELEMENT:
        #   XeREF: NODAL COORDINATES OF 2D REFERENCE ELEMENT
        #   XeintREF: NODAL COORDINATES OF INTERFACE IN 2D REFERENCE ELEMENT
        #   XemodREF: NODAL COORDINATES OF 2D REFERENCE ELEMENT WITH TESSELLATION
        #   TemodREF: CONNECTIVITY MATRIX OF 2D REFERENCE ELEMENT WITH TESSELLATION
        #   XgmodREF: GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT WITH TESSELLATION, MODIFIED QUADRATURE
        ### 2D PHYSICAL ELEMENT:
        #   Xe: NODAL COORDINATES OF 2D PHYSICAL ELEMENT 
        #   Xeint: NODAL COORDINATES OF INTERFACE IN 2D PHYSICAL ELEMENT
        #   Xemod: NODAL COORDINATES OF 2D PHYSICAL ELEMENT WITH TESSELLATION
        #   Temod: CONNECTIVITY MATRIX OF 2D PHYSICAL ELEMENT WITH TESSELLATION
        
        # IN ORDER TO COMPUTE THE MODIFIED QUADRATURE, WE NEED TO:
        #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
        #    2. PERFORM TESSELLATION ON THE REFERENCE ELEMENT -> OBTAIN NODAL COORDINATES OF REFERENCE SUBELEMENTS
        #    3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS 
        #    4. PERFORM TESSELLATION ON PHYSICAL ELEMENT """
        
        element = self.Tinter[interface,0]
        Xe = self.X[self.T[element,:],:]
        Xeint = self.InterfaceCoordinates[interface,:,:]
        
        # 1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT
        XeintREF = np.zeros([2,2])
        for i in range(2):
            XeintREF[i,:] = self.InverseMapping(Xeint[i,:], element)
        
        # 2. DO TESSELLATION ON REFERENCE ELEMENT
        XeREF = np.array([[1,0], [0,1], [0,0]])
        XemodREF, TemodREF = self.Tessellation(XeREF,XeintREF)
        Nsub = np.shape(TemodREF)[0]
        
        # 3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS 
        XgmodREF = np.zeros([Nsub*self.Ng2D,self.dim])
        for i in range(Nsub):
            for ig in range(self.Ng2D):
                XgmodREF[self.Ng2D*i+ig,:] = self.N[ig,:] @ XemodREF[TemodREF[i,:]]
                
        # 4. PERFORM TESSELLATION ON PHYSICAL ELEMENT
        Xemod, Temod = self.Tessellation(Xe,Xeint)
        
        # 5. DETERMINE ON TO WHICH REGION (INSIDE OR OUTSIDE) FALLS EACH SUBELEMENT
        if self.LevelSet[self.Tinter[interface,1]] < 0:  # COMMON NODE YIELD PHI < 0 -> INSIDE REGION
            Dom = np.array([-1,1,1])
        else:
            Dom = np.array([1,-1,-1])
        
        return Xemod, Temod, XgmodREF, Dom
    
    
    def ComputeModifiedQuadrature1D(self,interface):
        """ This function returns the modified 1D Gauss quadrature to integrate over the interface linear approximation cutting the element. 
        Input: -
        Output: - 
        
        Important quantities in this routine:
        ### 1D REFERENCE ELEMENT:
        #   zline: GAUSS NODAL COORDINATES IN 1D REFERENCE ELEMENT
        #   wline: GAUSS WEIGHTS IN 1D REFERENCE ELEMENT
        ### 2D REFERENCE ELEMENT:
        #   XeREF: NODAL COORDINATES OF 2D REFERENCE ELEMENT
        #   XeintREF: NODAL COORDINATES OF INTERFACE IN 2D REFERENCE ELEMENT
        #   XgintREF: GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT
        ### 2D PHYSICAL ELEMENT:
        #   Xe: NODAL COORDINATES OF 2D PHYSICAL ELEMENT 
        #   Xeint: NODAL COORDINATES OF INTERFACE IN 2D PHYSICAL ELEMENT
        #   Xgint: GAUSS NODAL COORDINATES IN 2D PHYSICAL ELEMENT 
         
        # IN ORDER TO COMPUTE THE MODIFIED QUADRATURE, WE NEED TO:
        #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
        #    2. MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE 
        # """
        
        element = self.Tinter[interface,0]
        Xeint = self.InterfaceCoordinates[interface,:,:]
        
        # 1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT
        XeintREF = np.zeros([2,2])
        for i in range(2):
            XeintREF[i,:] = self.InverseMapping(Xeint[i,:], element)
            
        # 2. MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE 
        XgintREF = np.zeros([self.Ng1D,self.dim])
        for ig in range(self.Ng1D):
            XgintREF[ig,:] = self.N1D[ig,:] @ XeintREF
        
        return XgintREF
    
    
    def AssembleGlobalSystem(self):
        """ This routine assembles the global matrices derived from the discretised linear system of equations used the common Galerkin approximation. 
        Nonetheless, due to the unfitted nature of the method employed, integration in cut cells (elements containing the interface between plasma region 
        and vacuum region, defined by the level-set 0-contour) must be treated accurately. """
        
        # INITIALISE GLOBAL SYSTEM MATRICES
        self.LHS = np.zeros([self.Nn,self.Nn])
        self.RHS = np.zeros([self.Nn, 1])
        
        # ELEMENTS INSIDE AND OUTSIDE PLASMA REGION (ELEMENTS WHICH ARE NOT CUT)
        print("     Assemble non-cut elements...", end="")
        for elem in np.concatenate((self.PlasmaElems, self.VacuumElems), axis=0): 
            #print(elem)   
            # ISOLATE PHYSICAL NODAL COORDINATES
            Xe = self.X[self.T[elem,:],:]
            Rmean = np.sum(Xe[:,0])/self.n   # mean elemental radial position
            # ISOLATE NODAL PHI VALUES
            PHIe = self.phi_inner0[self.T[elem,:]]
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(self.Ng2D):  
                # MAPP GAUSS NODAL COORDINATES FROM REFERENCE ELEMENT TO PHYSICAL ELEMENT
                Xg = self.N[ig,:] @ Xe
                # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL ELEMENT
                PHIg = self.N[ig,:] @ PHIe
                # COMPUTE JACOBIAN INVERSE AND DETERMINANT
                invJ, detJ = Jacobian(Xe[:,0],Xe[:,1],self.dNdxi[ig,:],self.dNdeta[ig,:])
                detJ *= 2*np.pi*Rmean   # ACCOUNT FOR AXISYMMETRICAL 
                # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
                if elem in self.PlasmaElems:
                    SourceTerm = self.mu0*Xg[0]*Jphi(self.mu0,Xg[0],Xg[1],PHIg)
                else:
                    SourceTerm = 0
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
                for i in range(self.n):   # ROWS ELEMENTAL MATRIX
                    for j in range(self.n):   # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) *(Jacobiano*2pi*rad) ]  
                        self.LHS[self.T[elem,i],self.T[elem,j]] -= (np.transpose((invJ@np.array([[self.dNdxi[ig,i]],[self.dNdeta[ig,i]]])))@(invJ@np.array([[self.dNdxi[ig,j]],[self.dNdeta[ig,j]]])))*detJ*self.Wg2D[ig]
                        ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j *(Jacobiano*2pi*rad) ]  ONLY RESPECT TO R
                        self.LHS[self.T[elem,i],self.T[elem,j]] -= (1/Xg[0])*self.N[ig,j] * (invJ[0,:]@np.array([[self.dNdxi[ig,i]],[self.dNdeta[ig,i]]]))*detJ*self.Wg2D[ig]
                    # COMPUTE RHS VECTOR TERMS [ (source term)*N_i*(Jacobiano *2pi*rad) ]
                    self.RHS[self.T[elem,i]] += SourceTerm * self.N[ig,i] *detJ*self.Wg2D[ig]
        
        print("Done!")
        
        print("     Assemble cut elements...", end="")
        # INTERFACE ELEMENTS (CUT ELEMENTS)
        for interface, elem in enumerate(self.InterElems):
            #print(interface, elem)
            # ISOLATE PHYSICAL NODAL COORDINATES
            Xe = self.X[self.T[elem,:],:]
            Rmean = np.sum(Xe[:,0])/self.n   # mean elemental radial position
            # ISOLATE NODAL PHI VALUES
            PHIe = self.phi_inner0[self.T[elem,:]]
            
            # NOW, EACH INTERFACE ELEMENT NEEDS TO BE DIVIDED INTO SUBELEMENTS ACCORDING TO THE POSITION OF THE APPROXIMATED INTERFACE,
            # ON EACH SUBELEMENT A MODIFIED QUADRATURE NEEDS TO BE BUILT IN ORDER TO INTEGRATE 
            Xemod, Temod, XgmodREF, Dom = self.ComputeModifiedQuadrature2D(interface)
            # EVALUATE 2D REFERENCE SHAPE FUNCTIONS ON MODIFIED GAUSS NODES
            Nmod, dNdximod, dNdetamod = EvaluateShapeFunctions(self.ElType, self.ElOrder, self.n, XgmodREF)
            # LOOP OVER SUBELEMENTS 
            Nsubelem = len(Temod[:,0])
            for subelem in range(Nsubelem):  
                # ISOLATE NODAL COORDINATES FOR SUBELEMENT
                Xesub = Xemod[Temod[subelem,:],:]
                Rmeansub = np.sum(Xesub[:,0])/self.n   # mean subelemental radial position
                # LOOP OVER GAUSS INTEGRATION NODES
                for igsub in range(self.Ng2D): 
                    ig = subelem*Nsubelem+igsub 
                    # MAPP GAUSS NODAL COORDINATES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    Xgmod = self.N[igsub,:] @ Xesub
                    # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PHIgmod = Nmod[ig,:] @ PHIe
                    # COMPUTE JACOBIAN INVERSE AND DETERMINANT
                    invJ, detJ = Jacobian(Xesub[:,0],Xesub[:,1],dNdximod[ig,:],dNdetamod[ig,:])
                    detJ *= 2*np.pi*Rmeansub   # ACCOUNT FOR AXISYMMETRICAL 
                    # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
                    if Dom[subelem] < 0:
                        SourceTerm = self.mu0*Xgmod[0]*Jphi(self.mu0,Xgmod[0],Xgmod[1],PHIgmod)
                    else:
                        SourceTerm = 0
                    # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
                    for i in range(self.n):   # ROWS ELEMENTAL MATRIX
                        for j in range(self.n):   # COLUMNS ELEMENTAL MATRIX
                            # COMPUTE LHS MATRIX TERMS
                            ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) *(Jacobiano*2pi*rad) ]  
                            self.LHS[self.T[elem,i],self.T[elem,j]] -= (np.transpose((invJ@np.array([[dNdximod[ig,i]],[dNdetamod[ig,i]]])))@(invJ@np.array([[dNdximod[ig,j]],[dNdetamod[ig,j]]])))*detJ*self.Wg2D[igsub]
                            ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j *(Jacobiano*2pi*rad) ]  ONLY RESPECT TO R
                            self.LHS[self.T[elem,i],self.T[elem,j]] -= (1/Xgmod[0])*Nmod[ig,j] * (invJ[0,:]@np.array([[dNdximod[ig,i]],[dNdetamod[ig,i]]]))*detJ*self.Wg2D[igsub]
                        # COMPUTE RHS VECTOR TERMS [ (source term)*N_i*(Jacobiano *2pi*rad) ]
                        self.RHS[self.T[elem,i]] += SourceTerm * Nmod[ig,i] *detJ*self.Wg2D[igsub]
        print("Done!")
        
        return
    
    def InterfaceNormal(self, Xe, Xeint, LSe):
        """ This function computes the interface normal vector pointing outwards. """
        
        dx = Xeint[1,0] - Xeint[0,0]
        dy = Xeint[1,1] - Xeint[0,1]
        ntest = np.array([-dy, dx])   # test this normal vector
        ntest = ntest/np.linalg.norm(ntest)   # normalize
        Xintmean = np.array([np.mean(Xeint[:,0]), np.mean(Xeint[:,1])])  # mean point on interface
        Xtest = Xintmean + 2*ntest  # physical point on which to test the Level-Set 
        
        # INTERPOLATE LEVEL-SET ON XTEST
        LStest = 0
        for i in range(self.n):
            LStest += ShapeFunctionsPhysical(Xtest, Xe, self.ElType, self.ElOrder, i+1)*LSe[i]
            
        # CHECK SIGN OF LEVEL-SET 
        if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
            n = ntest
        else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
            n = -ntest
    
        return n
    
    def BoundaryCondition(self, x):
        
        # ADIMENSIONALISE 
        xstar = x/self.R0
        phiD = xstar[0]**4/8 + self.coeffs[0] + self.coeffs[1]*xstar[0]**2 + self.coeffs[2]*(xstar[0]**4-4*xstar[0]**2*xstar[1]**2)
        
        return phiD

    
    def ApplyBoundaryConditions(self):
        """ Function computing the boundary integral terms arising from Nitsche's method (weak imposition of BC) and assembling 
        into the global system. Such terms only affect the elements containing the interface. """
        
        for interface, elem in enumerate(self.InterElems):
            # ISOLATE PHYSICAL NODAL COORDINATES
            Xe = self.X[self.T[elem,:],:]
            # ISOLATE NODAL LEVEL-SET VALUES
            LSe = self.LevelSet[self.T[elem,:]]
            # ISOLATE INTERFACE LINEAR APPROXIMATION 
            Xeint = self.InterfaceCoordinates[interface,:,:]
            # COMPUTE NORMAL VECTOR RESPECT TO INTERFACE
            Nvec = self.InterfaceNormal(Xe, Xeint, LSe)
            # COMPUTE 1D GAUSS MODIFED QUADRATURE ON REFERENCE INTERFACE 
            XgintREF = self.ComputeModifiedQuadrature1D(interface)
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON MODIFIED QUADRATURE GAUSS NODES
            Nmod, dNdximod, dNdetamod = EvaluateShapeFunctions(self.ElType, self.ElOrder, self.n, XgintREF)
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(self.Ng1D):  
                # MAPP 1D GAUSS NODAL COORDINATES ON PHYSICAL INTERFACE 
                Xg = self.N1D[ig,:] @ Xeint
                # COMPUTE BOUNDARY CONDITION VALUES
                PHId = self.BoundaryCondition(Xg)
                # COMPUTE JACOBIAN OF TRANSFORMATION
                detJ1D = Jacobian1D(Xeint[:,0],Xeint[:,1],self.dNdxi1D[ig,:])
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                for i in range(self.n):  # ROWS ELEMENTAL MATRIX
                    for j in range(self.n):  # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) *(Jacobiano*2pi*rad) ]  
                        self.LHS[self.T[elem,i],self.T[elem,j]] += Nmod[ig,i] * Nvec @ np.array([[dNdximod[ig,j]],[dNdetamod[ig,j]]]) * detJ1D * self.Wg1D[ig]
                        ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) *(Jacobiano*2pi*rad) ]
                        self.LHS[self.T[elem,i],self.T[elem,j]] += Nvec @ np.array([[dNdximod[ig,i]],[dNdetamod[ig,i]]]) * Nmod[ig,j] * detJ1D * self.Wg1D[ig]
                        ### PENALTY TERM   [ beta * (N_i*N_j) *(Jacobiano*2pi*rad) ]
                        self.LHS[self.T[elem,i],self.T[elem,j]] += self.beta * Nmod[ig,i] * Nmod[ig,j] * detJ1D * self.Wg1D[ig]
                    # COMPUTE RHS VECTOR TERMS 
                    ### SYMMETRIC NITSCHE'S METHOD TERM  [ Phi_D * (n dot nabla(N_i)) * (Jacobiano *2pi*rad) ]
                    self.RHS[self.T[elem,i]] +=  PHId * Nvec @ np.array([[dNdximod[ig,i]],[dNdetamod[ig,i]]]) * detJ1D * self.Wg1D[ig]
                    ### PENALTY TERM   [ beta * N_i * Phi_D *(Jacobiano*2pi*rad) ]
                    self.RHS[self.T[elem,i]] +=  self.beta * Nmod[ig,i] * PHId * detJ1D * self.Wg1D[ig]
                
        return
    
    def SolveSystem(self):
        
        self.phi_inner1 = np.linalg.solve(self.LHS, self.RHS)
        
        return
    
    def CheckConvergence(self,loop):
        
        if loop == "INNER":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.phi_inner1) > 0:
                L2residu = np.linalg.norm(self.phi_inner1 - self.phi_inner0)/np.linalg.norm(self.phi_inner1)
            else: 
                L2residu = np.linalg.norm(self.phi_inner1 - self.phi_inner0)
            if L2residu < self.TOL_inner:
                self.marker_inner = False   # STOP WHILE LOOP 
                self.phi_outer1 = self.phi_inner1
            else:
                self.marker_inner = True
                self.phi_inner0 = self.phi_inner1
            
        elif loop == "OUTER":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.phi_outer1) > 0:
                L2residu = np.linalg.norm(self.phi_outer1 - self.phi_outer0)/np.linalg.norm(self.phi_outer1)
            else: 
                L2residu = np.linalg.norm(self.phi_outer1 - self.phi_outer0)
            if L2residu < self.TOL_outer:
                self.marker_outer = False   # STOP WHILE LOOP 
                self.phi_converged = self.phi_outer1
            else:
                self.marker_outer = True
                self.phi_outer0 = self.phi_outer1
        return 

    
    def PlasmaEquilibrium(self):
        
        # INITIALISE VARIABLES
        self.phi_inner0 = np.zeros([self.Nn])      # solution at inner iteration n
        self.phi_inner1 = np.zeros([self.Nn])      # solution at inner iteration n+1
        self.phi_outer0 = np.zeros([self.Nn])      # solution at outer iteration n
        self.phi_outer1 = np.zeros([self.Nn])      # solution at outer iteration n+1
        self.phi_converged = np.zeros([self.Nn])   # solution at outer iteration n+1
        
        # INITIAL GUESS FOR MAGNETIC FLUX
        print("COMPUTE INITIAL GUESS...", end="")
        self.phi_inner0 = self.InitialGuess()
        self.phi_outer0 = self.phi_inner0
        print('Done!')
        
        # INITIALISE LEVEL-SET FUNCTION
        print("INITIALISE LEVEL-SET...", end="")
        self.LevelSet = self.phi_inner0
        print('Done!')
        
        # CLASSIFY ELEMENTS  ->  OBTAIN PLASMAELEMS, VACUUMELEMS, INTERELEMS
        print("CLASSIFY ELEMENTS...", end="")
        self.ClassifyElements()
        print("Done!")
        
        # COMPUTE INTERFACE LINEAR APPROXIMATION
        print("APPROXIMATE INTERFACE...", end="")
        self.ComputeInterfaceCoordinates()
        print("Done!")
        
        self.PlotSolution(self.phi_inner0)
        
        # COMPUTE NUMERICAL INTEGRATION ELEMENTS: NUMERICAL QUADRATURE AND SHAPE FUNCTIONS EVALUATED AT INTEGRATION NODES
        # OBTAIN GAUSS QUADRATURE FOR NUMERICAL INTEGRATION
        print('COMPUTE NUMERICAL INTEGRATION QUADRATURE AND SHAPE FUNCTIONS...', end="")
        QuadratureOrder = 2
        #### QUADRATURE TO INTEGRATE SURFACES (2D)
        self.Zg2D, self.Wg2D, self.Ng2D = GaussQuadrature(self.ElType,QuadratureOrder)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.Zg1D, self.Wg1D, self.Ng1D = GaussQuadrature(2,QuadratureOrder)
        
        # EVALUATE SHAPE FUNCTIONS AT GAUSS NODES
        #### QUADRATURE TO INTEGRATE SURFACES (2D)
        self.N, self.dNdxi, self.dNdeta = EvaluateShapeFunctions(self.ElType, self.ElOrder, self.n, self.Zg2D)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.n1D = 2
        self.N1D, self.dNdxi1D, foo = EvaluateShapeFunctions(2, QuadratureOrder-1, self.n1D, self.Zg1D)
        print('Done!')
        
        
        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.marker_outer = True
        self.it_outer = 0
        while (self.marker_outer == True and self.it_outer < self.itmax):
            self.it_outer += 1
            self.marker_inner = True
            self.it_inner = 0
            while (self.marker_inner == True and self.it_inner < self.itmax):
                self.it_inner += 1
                self.AssembleGlobalSystem()
                self.ApplyBoundaryConditions()
                self.SolveSystem()
                print('OUTER ITERATION = '+str(self.it_outer)+' , INNER ITERATION = '+str(self.it_inner))

                self.PlotSolution(self.phi_inner1)
                
                self.CheckConvergence("INNER")
                
                
            self.CheckConvergence("OUTER")
            
                
        return
    
    def PlotSolution(self,phi):
        if len(np.shape(phi)) == 2:
            phi = phi[:,0]
        plt.figure(figsize=(7,10))
        plt.tricontourf(self.X[:,0],self.X[:,1], phi, levels=30)
        #plt.tricontour(self.X[:,0],self.X[:,1], phi, levels=[0], colors='k')
        plt.colorbar()

        plt.show()
        return
    
    def PlotMesh(self):
        Tmesh = self.T + 1
        # Plot nodes
        plt.figure(figsize=(7,10))
        plt.plot(self.X[:,0],self.X[:,1],'.')
        for e in range(self.Ne):
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='black', linewidth=1)
        plt.show()
        return
    
    def PlotMeshClassifiedElements(self):
        plt.figure(figsize=(7,10))
        plt.tricontourf(self.X[:,0],self.X[:,1], self.phi_inner0, levels=30, cmap='plasma')
        plt.tricontour(self.X[:,0],self.X[:,1], self.phi_inner0, levels=[0], colors='k')
        #plt.colorbar()

        # PLOT NODES
        plt.plot(self.X[:,0],self.X[:,1],'.',color='black')
        Tmesh = self.T +1
        # PLOT PLASMA REGION ELEMENTS
        for e in self.PlasmaElems:
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='red', linewidth=1)
        # PLOT VACCUM ELEMENTS
        for e in self.VacuumElems:
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='black', linewidth=1)
        # PLOT INTERFACE ELEMENTS
        for e in self.InterElems:
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='yellow', linewidth=1)
                    
        plt.show()
        return