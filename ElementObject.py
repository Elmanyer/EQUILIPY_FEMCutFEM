""" THIS SCRIPT DEFINES THE ATTRIBUTES AND METHODS CORRESPONDING TO OBJECT ELEMENT. THIS CLASS SHALL GATHER ALL THE RELEVANT INFORMATION (COORDINATES,
QUADRATURES, NODAL VALUES...) FOR A SINGLE ELEMENT IN THE MESH. """

from GaussQuadrature import *
from ShapeFunctions import *
from scipy import optimize

class Element:
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,LSe,PHIe):
        
        self.index = index                                  # GLOBAL INDEX ON COMPUTATIONAL MESH
        self.ElType = ElType                                # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
        self.ElOrder = ElOrder                              # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.n = ElementalNumberOfNodes(ElType, ElOrder)    # NUMBER OF NODES PER ELEMENT
        self.n1D = ElOrder+1                                # NUMBER OF NODES PER EDGE
        self.Xe = Xe                                        # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.dim = len(Xe[0,:])                             # SPATIAL DIMENSION
        self.Te = Te                                        # ELEMENTAL CONNECTIVITIES
        self.LSe = LSe                                      # ELEMENTAL NODAL LEVEL-SET VALUES
        self.PHIe = PHIe                                    # ELEMENTAL NODAL PHI VALUES
        self.Dom = None                                     # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; +1: "VACUUM" ; 0: "INTERFACE")
        
        # STANDARD ELEMENTAL INTEGRATION QUADRATURES (1D AND 2D)
        self.Ng1D = None            # NUMBER OF GAUSS NODES IN STANDARD 1D GAUSS QUADRATURE
        self.Wg1D = None            # STANDARD 1D GAUSS INTEGRATION WEIGTHS
        self.N1D = None             # REFERENCE 1D SHAPE FUNCTIONS EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES 
        self.dNdxi1D = None         # REFERENCE 1D SHAPE FUNCTIONS DERIVATIVES EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES 
        self.Ng2D = None            # NUMBER OF GAUSS NODES IN STANDARD 2D GAUSS QUADRATURE
        self.Wg2D = None            # STANDARD 2D GAUSS INTEGRATION WEIGTHS
        self.N = None               # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.dNdxi = None           # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.dNdeta = None          # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        
        ### ATTRIBUTES FOR INTERFACE ELEMENTS
        self.interface = None       # INTERFACE GLOBAL INDEX
        self.Xeint = None           # PHYSICAL INTERFACE NODAL COORDINATES 
        self.permu = None           # PERMUTATIONS IN THE CONECTIVITY SO THAT THE COMMON NODE IS FIRST
        self.Nsub = None            # NUMBER OF SUBELEMENTS GENERATED IN TESSELLATION
        self.Xemod = None           # MODIFIED ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.Temod = None           # MODIFIED CONNECTIVITY MATRIX FOR TESSELLED INTERFACE PHYSICAL ELEMENT
        self.Dommod = None          # DOMAIN WHERE EACH SUBELEMENT AFTER TESSELLATION LIES ("PLASMA" OR "VACUUM")
        self.XimodREF = None        # MODIFIED ELEMENTAL NODAL MATRIX (REFERENCE COORDINATES)
        self.TemodREF = None        # MODIFIED CONNECTIVITY MATRIX FOR TESSELLED INTERFACE REFERENCE ELEMENT
        self.NormalVec = None       # INTERFACE NORMAL VECTOR POINTING OUTWARDS
        
        # MODIFIED ELEMENTAL INTEGRATION QUADRATURES (1D AND 2D)
        self.XigintREF = None       # MODIFIED REFERENCE GAUSS INTEGRATION NODES COMPUTED FROM 1D STANDARD QUADRATURE
        self.XigmodREF = None       # MODIFIED REFERENCE GAUSS INTEGRATION NODES COMPUTED FROM 2D STANDARD QUADRATURE
        
        self.Nintmod = None         # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.dNdxiintmod = None     # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.dNdetaintmod = None    # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES
        self.Nmod = None            # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT MODIFIED 2D GAUSS INTEGRATION NODES
        self.dNdximod = None        # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT MODIFIED 2D GAUSS INTEGRATION NODES
        self.dNdetamod = None       # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT MODIFIED 2D GAUSS INTEGRATION NODES
        
        return
    
    def Mapping(self,Xi):
        """ This function implements the mapping corresponding to the transformation from natural to physical coordinates. 
        That is, given a point in the reference element with coordinates Xi, this function returns the coordinates X of the corresponding point mapped
        in the physical element with nodal coordinates Xe. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric equations. 
        
        Input: - Xg: coordinates of point in reference space for which to compute the coordinate in physical space
               - Xe: nodal coordinates of physical element
        Output: - X: coodinates of mapped point in reference element """
        
        N, foo, foo = EvaluateReferenceShapeFunctions(Xi, self.ElType, self.ElOrder, self.n)
        X = N @ self.Xe
        return X
    
    def InverseMapping(self, X):
        """ This function implements the inverse mapping corresponding to the transformation from natural to physical coordinates (thus, for the inverse transformation
        we go from physical to natural coordinates). That is, given a point in physical space with coordinates X in the element with nodal coordinates Xe, 
        this function returns the point mapped in the reference element with natural coordinates Xi. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric equations. 
        
        Input: - X: physical coordinates of point for which compute the corresponding point in the reference space
               - Xe: nodal coordinates of physical element
        Output: - Xg: coodinates of mapped point in reference element """
        
        # DEFINE THE NONLINEAR SYSTEM 
        def fun(Xi, X, Xe):
            f = np.array([-X[0],-X[1]])
            for i in range(self.n):
                Nig, foo, foo = ShapeFunctionsReference(Xi, self.ElType, self.ElOrder, i+1)
                f[0] += Nig*Xe[i,0]
                f[1] += Nig*Xe[i,1]
            return f
        # SOLVE NONLINEAR SYSTEM
        Xi0 = np.array([1/2, 1/2])  # INITIAL GUESS FOR ROOT SOLVER
        sol = optimize.root(fun, Xi0, args=(X,self.Xe))
        Xi = sol.x
        return Xi
    
    
    def InterfaceLinearApproximation(self):
        """ Function computing the intersection points between the element edges and the interface (for elements containing the interface) 
            FOR THE MOMENT, DESIGNED EXCLUSIVELY FOR TRIANGULAR ELEMENTS
        Input: - element_index: index of element for which to compute approximated interface coordinates 
        Output: - Xeint: matrix containing the coordinates of points located at the intrsection between the interface and the element's edges
                - modified nodal coordinate matrix
                - Te: modified elemental conectivity matrix, where the first entry corresponds to the node "alone" in its respective region (common node 
                    to both edges intersecting with interface), and the following entries correspond to the other elemental nodes, which together with the
                    first one, define the edges intersecting the interface. 
                - pos: vector storing the permutations done on the original conectivity in order to place the common node first. """
        
        # READ NODAL COORDINATES 
        Xe = self.Xe
        # READ LEVEL-SET NODAL VALUES
        LSe = self.LSe  
        # LOOK FOR THE NODE WHICH HAS DIFFERENT SIGN...
        pospos = np.where(LSe > 0)[0]
        posneg = np.where(LSe < 0)[0]
        # ... PIVOT COORDINATES MATRIX ACCORDINGLY
        if len(pospos) > len(posneg):  # 2 nodal level-set values are positive (outside plasma region)
            pos = np.concatenate((posneg,pospos),axis=0)
            Xe = Xe[pos]
            LSe = LSe[pos]
        else: # 2 nodal level-set values are negative (inside plasma region)
            pos = np.concatenate((pospos,posneg),axis=0)
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
        self.Xeint = np.zeros([2,2])
        for i, edge in enumerate([1,2]):
            sol = optimize.root(fun, Xe[0,0], args=(Xe,LSe,edge))
            self.Xeint[i,:] = [sol.x, z(sol.x,Xe,edge)]
            
        self.Xemod = np.concatenate((Xe,self.Xeint), axis = 0)
        self.permu = pos
        
        return 
    
    @staticmethod
    def CheckNodeOnEdge(x,Xe,TOL):
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
        
    
    @staticmethod
    def Tessellation(Mode,**kwargs):
        """ This function performs the TESSELLATION of an element with nodal coordinates Xe and interface coordinates Xeint (intersection with edges) 
        Input: - Mode: for Mode=0 we pass the input (Xe,Xeint); for Mode=1 we pass input (Xemod) 
               - Xe: element nodal coordinates 
               - Xeint: coordinates of intersection points between interface and edges 
               - Xemod: modified nodal coordinate matrix, where the first row is the common node to both edge intersecting the interface
        Output: - TeTESS: Tessellation connectivity matrix such that 
                        TeTESS = [[Connectivities for subelement 1]
                                  [Connectivities for subelement 2]
                                                ...          
                - for Mode=0, XeTESS is outputed as well. XeTESS: Nodal coordinates matrix storing the coordinates of the element vertex and interface points                   
                """
                
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        # HOWEVER, WHEN LOOKING FOR THE LINEAR APPROXIMATION OF THE PHYSICAL INTERFACE THIS PROCESS IS ALREADY DONE, THEREFORE WE CAN SKIP IT. 
        # IF INPUT Xemod IS PROVIDED, THE TESSELLATION IS DONE ACCORDINGLY TO MODIFIED NODAL MATRIX Xemod WHICH IS ASSUMED TO HAS THE PREVIOUSLY DESCRIBED STRUCTURE.
        # IF NOT, THE COMMON NODE IS DETERMINED (THIS IS THE CASE FOR INSTANCE WHEN THE REFERENCE ELEMENT IS TESSELLATED).
                
        if Mode == 1:
            XeTESS = kwargs['Xemod']
        elif Mode == 0:
            Xe = kwargs['Xe']
            Xeint = kwargs['Xeint']
            Nint = np.shape(Xeint)[0]  # number of intersection points
            edgenodes = np.zeros(np.shape(Xeint), dtype=int)
            nodeedgeinter = np.zeros([Nint], dtype=int)
            for i in range(Nint):
                edgenodes[i,:] = Element.CheckNodeOnEdge(Xeint[i,:],Xe,1e-4)
            commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop()
            for i in range(Nint):
                edgenodesset = set(edgenodes[i,:])
                edgenodesset.remove(commonnode)
                nodeedgeinter[i] = edgenodesset.pop()
        
            Xe = Xe[np.concatenate((np.array([commonnode]), nodeedgeinter), axis=0),:]
            # MODIFIED NODAL MATRIX AND CONECTIVITIES, ACCOUNTING FOR 3 SUBTRIANGLES 
            XeTESS = np.concatenate((Xe, Xeint), axis=0)
            
        # ONCE THE NODAL MATRIX IS ORGANISED, THE CONNECTIVITIES ARE TRIVIAL AND CAN BE HARD-CODED 
        TeTESS = np.zeros([3, 3], dtype = int)  # connectivities for 3 subtriangles
        TeTESS[0,:] = [0, 3, 4]  # first triangular subdomain is common node and intersection nodes

        # COMPARE DISTANCE INTERFACE-(EDGE NODE)
        edge = 1
        distance1 = np.linalg.norm(XeTESS[edge,:]-XeTESS[edge+2,:])
        edge = 2
        distance2 = np.linalg.norm(XeTESS[edge,:]-XeTESS[edge+2,:])

        if distance1 <= distance2:
            TeTESS[1,:] = [3, 1, 2]
            TeTESS[2,:] = [3, 4, 2]
        if distance1 > distance2:
            TeTESS[1,:] = [4, 2, 1]
            TeTESS[2,:] = [4, 3, 1]
        
        if Mode == 1:
            return TeTESS
        elif Mode == 0:
            return XeTESS, TeTESS
        
    
    
    def ComputeModifiedQuadratures(self,Order):
        """ This function computes the Gauss quadratures corresponding to a 2D and 1D integration. In the case of elements which 
        are NOT cut by the interface, the standard FEM numerical integration quadratures are prepared. On the other hand, if the element
        IS cut by the interface, an adapted quadrature is computed in this case by modifying the standard approach.  
        Input: - Order: Gauss quadrature order
        
        Important quantities:
            ### 1D REFERENCE ELEMENT:
            #   Xig1D: STANDARD GAUSS NODAL COORDINATES IN 1D REFERENCE ELEMENT
            #   Wg1D: STANDARD GAUSS WEIGHTS IN 1D REFERENCE ELEMENT
            ### 2D REFERENCE ELEMENT:
            #   Xig2D: STANDARD GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT
            #   Wg2D: STANDARD GAUSS WEIGHTS IN 2D REFERENCE ELEMENT
            
            #   XiREF: NODAL COORDINATES OF 2D REFERENCE ELEMENT
            #   XiintREF: INTERFACE NODAL COORDINATES IN 2D REFERENCE ELEMENT
            #   XigintREF: INTERFACE GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT, MODIFIED 1D QUADRATURE
            
            #   XimodREF: NODAL COORDINATES OF 2D REFERENCE ELEMENT WITH TESSELLATION
            #   TemodREF: CONNECTIVITY MATRIX OF 2D REFERENCE ELEMENT WITH TESSELLATION
            #   XigmodREF: GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT WITH TESSELLATION, MODIFIED 2D QUADRATURE
            ### 2D PHYSICAL ELEMENT:
            #   Xe: NODAL COORDINATES OF 2D PHYSICAL ELEMENT 
            #   Xeint: NODAL COORDINATES OF INTERFACE IN 2D PHYSICAL ELEMENT
            #   Xgint: GAUSS NODAL COORDINATES IN 2D PHYSICAL ELEMENT 
            #   Xemod: NODAL COORDINATES OF 2D PHYSICAL ELEMENT WITH TESSELLATION
            #   Temod: CONNECTIVITY MATRIX OF 2D PHYSICAL ELEMENT WITH TESSELLATION
            
            # IN ORDER TO COMPUTE THE 1D MODIFIED QUADRATURE, WE NEED TO:
            #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
            #    2. MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE
            
            # IN ORDER TO COMPUTE THE 2D MODIFIED QUADRATURE, WE NEED TO:
            #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
            #    2. PERFORM TESSELLATION ON THE REFERENCE ELEMENT -> OBTAIN NODAL COORDINATES OF REFERENCE SUBELEMENTS
            #    3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS 
            #    4. PERFORM TESSELLATION ON PHYSICAL ELEMENT 
            #    5. DETERMINE ON TO WHICH REGION (INSIDE OR OUTSIDE) FALLS EACH SUBELEMENT
        """
        
        # FOR ALL ELEMENTS, COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE, FOR BOTH 1D AND 2D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES (2D)
        self.Xig2D, self.Wg2D, self.Ng2D = GaussQuadrature(self.ElType,Order)
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        self.Xig1D, self.Wg1D, self.Ng1D = GaussQuadrature(0,Order)
        
        # FOR ALL ELEMENTS, WE EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        self.N, self.dNdxi, self.dNdeta = EvaluateReferenceShapeFunctions(self.Xig2D, self.ElType, self.ElOrder, self.n)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.N1D, self.dNdxi1D, foo = EvaluateReferenceShapeFunctions(self.Xig1D, 0, Order-1, self.n1D)
        
        # THE REFERENCE SHAPE FUNCTIONS EVALUATED ON THE STANDARD GAUSS INTEGRATION NODES REPRESENTS THE STANDARD QUADRATURE FOR VACUUM AND PLASMA REGION ELEMENTS,
        # BUT THEY ARE ALSO NECESSARY TO COMPUTE THE MODIFIED QUADRATURES ON THE INTERFACE ELEMENTS
            
        if self.Dom == 0:   # FOR INTERFACE ELEMENTS, THE MODIFIED GAUSS INTEGRATION NODES MUST BE COMPUTED
            
            # 1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT
            XeintREF = np.zeros(np.shape(self.Xeint))
            for i in range(2):
                XeintREF[i,:] = self.InverseMapping(self.Xeint[i,:])
                
            ######### 1D MODIFIED GAUSS NODES
            # 2(1D). MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE 
            self.XigintREF = np.zeros([self.Ng1D,self.dim])
            for ig in range(self.Ng1D):
                self.XigintREF[ig,:] = self.N1D[ig,:] @ XeintREF
            #########
            
            ######### 2D MODIFIED GAUSS NODES
            # 2(2D). DO TESSELLATION ON REFERENCE ELEMENT
            XeREF = np.array([[1,0], [0,1], [0,0]])
            self.XimodREF, self.TemodREF = self.Tessellation(Mode=0,Xe=XeREF,Xeint=XeintREF)
            self.Nsub = np.shape(self.TemodREF)[0]
            
            # 3(2D). MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS 
            self.XigmodREF = np.zeros([self.Nsub*self.Ng2D,self.dim])
            for i in range(self.Nsub):
                for ig in range(self.Ng2D):
                    self.XigmodREF[self.Ng2D*i+ig,:] = self.N[ig,:] @ self.XimodREF[self.TemodREF[i,:]]
                    
            # 4(2D). PERFORM TESSELLATION ON PHYSICAL ELEMENT
            self.Temod = self.Tessellation(Mode=1,Xemod=self.Xemod)
            
            # 5(2D). DETERMINE ON TO WHICH REGION (INSIDE OR OUTSIDE) FALLS EACH SUBELEMENT
            if self.LSe[self.permu[0]] < 0:  # COMMON NODE YIELD LS < 0 -> INSIDE REGION
                self.Dommod = np.array([-1,1,1])
            else:
                self.Dommod = np.array([1,-1,-1])
            #########
            
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON MODIFIED GAUSS INTEGRATION NODES
            self.Nintmod, self.dNdxiintmod, self.dNdetaintmod = EvaluateReferenceShapeFunctions(self.XigintREF, self.ElType, self.ElOrder, self.n)
            self.Nmod, self.dNdximod, self.dNdetamod = EvaluateReferenceShapeFunctions(self.XigmodREF, self.ElType, self.ElOrder, self.n)
        return 
    
    
    def InterfaceNormal(self):
        """ This function computes the interface normal vector pointing outwards. """
        
        dx = self.Xeint[1,0] - self.Xeint[0,0]
        dy = self.Xeint[1,1] - self.Xeint[0,1]
        ntest = np.array([-dy, dx])   # test this normal vector
        ntest = ntest/np.linalg.norm(ntest)   # normalize
        Xintmean = np.array([np.mean(self.Xeint[:,0]), np.mean(self.Xeint[:,1])])  # mean point on interface
        Xtest = Xintmean + 2*ntest  # physical point on which to test the Level-Set 
        
        # INTERPOLATE LEVEL-SET ON XTEST
        LStest = 0
        for i in range(self.n):
            LStest += ShapeFunctionsPhysical(Xtest, self.Xe, self.ElType, self.ElOrder, i+1)*self.LSe[i]
            
        # CHECK SIGN OF LEVEL-SET 
        if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
            self.NormalVec = ntest
        else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
            self.NormalVec = -ntest
        
        return 
        
    
def ElementalNumberOfNodes(elemType, elemOrder):
    # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
    match elemType:
        case 0:
            n = elemOrder +1
                
        case 1:
            match elemOrder:
                case 1:
                    n = 3
                case 2: 
                    n = 6
                case 3:
                    n = 10
        case 2:
            match elemOrder:
                case 1:
                    n = 4
                case 2:
                    n = 9
                case 3:
                    n = 16
    return n
    