""" THIS SCRIPT DEFINES THE ATTRIBUTES AND METHODS CORRESPONDING TO OBJECT ELEMENT. THIS CLASS SHALL GATHER ALL THE RELEVANT INFORMATION (COORDINATES,
QUADRATURES, NODAL VALUES...) FOR A SINGLE ELEMENT IN THE MESH. """

from src.GaussQuadrature import *
from src.ShapeFunctions import *
from scipy import optimize

class Element:
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,LSe):
        
        self.index = index                                              # GLOBAL INDEX ON COMPUTATIONAL MESH
        self.ElType = ElType                                            # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
        self.ElOrder = ElOrder                                          # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.n, self.nedge = ElementalNumberOfNodes(ElType, ElOrder)    # NUMBER OF NODES PER ELEMENT, PER ELEMENTAL EDGE
        self.Xe = Xe                                                    # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.dim = len(Xe[0,:])                                         # SPATIAL DIMENSION
        self.Te = Te                                                    # ELEMENTAL CONNECTIVITIES
        self.LSe = LSe                                                  # ELEMENTAL NODAL LEVEL-SET VALUES
        self.PHIe = np.zeros([self.n])                                  # ELEMENTAL NODAL PHI VALUES
        self.Dom = None                                                 # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; +1: "VACUUM" ; 0: "INTERFACE")
        
        # INTEGRATION QUADRATURES ENTITIES (1D AND 2D)
        self.Ng1D = None            # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 1D QUADRATURE
        self.XIg1D = None           # STANDARD 1D GAUSS INTEGRATION NODES 
        self.Wg1D = None            # STANDARD 1D GAUSS INTEGRATION WEIGTHS 
        self.N1D = None             # REFERENCE 1D SHAPE FUNCTIONS EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES 
        self.dNdxi1D = None         # REFERENCE 1D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES
        self.Ng2D = None            # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 2D GAUSS QUADRATURE
        self.XIg2D = None           # STANDARD 2D GAUSS INTEGRATION NODES 
        self.Wg2D = None            # STANDARD 2D GAUSS INTEGRATION WEIGTHS
        self.N = None               # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES 
        self.dNdxi = None           # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.dNdeta = None          # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.Xg2D = None            # PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM 2D REFERENCE ELEMENT
        self.invJg = None           # INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES 
        
        ### ATRIBUTES FOR INTERFACE AND BOUNDARY ELEMENTS
        self.NormalVec = None       # INTERFACE/BOUNDARY EDGE NORMAL VECTOR POINTING OUTWARDS
        
        ### ATTRIBUTES FOR BOUNDARY ELEMENTS
        self.PHI_Be = None           # PHI VALUE ON BOUNDARY EDGES NODES
        self.Nebound = None          # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.Tebound = None          # BOUNDARY EDGES CONNECTIVITY MATRIX (LOCAL INDEXES) 
        self.boundary = None         # GLOBAL INDEX FOR COMPUTATIONAL DOMAIN'S BOUNDARY ELEMENTAL EDGE
        self.XIgbound = None         # GAUSS INTEGRATION NODES ON BOUNDARY EDGE 
        self.Nbound = None           # REFERENCE 2D SHAPE FUNCTIONS EVALUATED BOUNDARY EDGE GAUSS INTEGRATION NODES
        self.dNdxibound = None       # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED BOUNDARY EDGE GAUSS INTEGRATION NODES
        self.dNdetabound = None      # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED BOUNDARY EDGE GAUSS INTEGRATION NODES
        self.Xgbound = None          # PHYSICAL 2D GAUSS INTEGRATION NODES MAPPED ON BOUNDARY EDGE
        self.detJgbound = None       # MATRIX DETERMINANTS OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL BOUNDARY EDGE EVALUATED AT GAUSS INTEGRATION NODES 
        
        ### ATTRIBUTES FOR INTERFACE ELEMENTS
        self.interface = None       # INTERFACE GLOBAL INDEX
        self.Xeint = None           # PHYSICAL INTERFACE NODAL COORDINATES 
        self.permu = None           # PERMUTATIONS IN THE CONECTIVITY SO THAT THE COMMON NODE IS FIRST
        self.Nsub = None            # NUMBER OF SUBELEMENTS GENERATED IN TESSELLATION
        self.Xemod = None           # MODIFIED ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.Temod = None           # MODIFIED CONNECTIVITY MATRIX FOR TESSELLED INTERFACE PHYSICAL ELEMENT
        # MODIFIED QUADRATURE FOR INTEGRATION ALONG INTERFACE 
        self.XIgint = None       # MODIFIED GAUSS INTEGRATION NODES COMPUTED FROM 1D STANDARD QUADRATURE 
        self.Nint = None         # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.Xgint = None        # PHYSICAL 2D GAUSS INTEGRATION NODES MAPPED FROM 1D REFERENCE ELEMENT
        self.dNdxiint = None     # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.dNdetaint = None    # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES
        self.detJgint = None     # MATRIX DETERMINANTS OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT INTERFACE EVALUATED AT GAUSS INTEGRATION NODES
        
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
    
    def ElementalInterpolation(self,X,Fe):
        """ Interpolate field F with nodal values Fe on point X using elemental shape functions. """
        F = 0
        for i in range(self.n):
            N = ShapeFunctionsPhysical(X, self.Xe, self.ElType, self.ElOrder, i+1)
            F += N*Fe[i]
        return F
    
    
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
        
        # OBTAIN INTERSECTION COORDINATES FOR EACH EDGE:
        self.Xeint = np.zeros([2,2])
        for i, edge in enumerate([1,2]):
            
            if np.abs(Xe[edge,0]-Xe[0,0]) > 1e-8:  # EDGE IS NOT VERTICAL
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

                # SOLVE TRANSCENDENTAL EQUATION AND COMPUTE INTERSECTION COORDINATES
                sol = optimize.root(fun, Xe[0,0], args=(Xe,LSe,edge))
                self.Xeint[i,:] = [sol.x, z(sol.x,Xe,edge)]
                
            else:  # IF THE ELEMENT'S EDGE IS VERTICAL
                r = Xe[0,0] 
                
                def fun(z,r,Xe,LSe,edge):
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
                    f = N0(r,z,Xe)*LSe[0] + Nedge(r,z,Xe,edge)*LSe[edge]
                    return f
                
                # SOLVE TRANSCENDENTAL EQUATION AND COMPUTE INTERSECTION COORDINATES
                sol = optimize.root(fun, Xe[0,1], args=(r,Xe,LSe,edge))
                self.Xeint[i,:] = [r, sol.x]
                
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
            else:
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
        Input: - Mode: for Mode="REFERENCE" we pass the input (Xe,Xeint,shortedge); for Mode="PHYSICAL" we pass input (Xemod) 
               - Xe: element nodal coordinates 
               - Xeint: coordinates of intersection points between interface and edges 
               - shortedge: edge for which the distance between common node and interface intersection is shortest IN THE PHYSICAL ELEMENT 
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
                
        if Mode == "PHYSICAL":
            XeTESS = kwargs['Xemod']
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
                shortedge = 1
            if distance1 > distance2:
                TeTESS[1,:] = [4, 2, 1]
                TeTESS[2,:] = [4, 3, 1]
                shortedge = 2
                
            return TeTESS, shortedge
                
        elif Mode == "REFERENCE":
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

            if kwargs["shortedge"] == 1:
                TeTESS[1,:] = [3, 1, 2]
                TeTESS[2,:] = [3, 4, 2]
            if kwargs["shortedge"] == 2:
                TeTESS[1,:] = [4, 2, 1]
                TeTESS[2,:] = [4, 3, 1]
                
            return XeTESS, TeTESS
        
        
    def FindBoundaryEdges(self,Tbound):
        """ This function finds the edges lying on the computational domain's boundary, for an element on the boundary. The different elemental attributes are set-up accordingly.
        
        Input: - Tbound: # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        
        Relevant attributes: 
            #   boundary: INDEX FOR COMPUTATIONAL DOMAIN'S BOUNDARY 
            #   Nebound: NUMBER OF BOUNDARY EDGES IN ELEMENT
            #   Tebound: BOUNDARY EDGES CONNECTIVITY MATRIX (LOCAL INDEXES) 
                """
        
        # LOOK WHICH BOUNDARIES ARE ASSOCIATED TO THE ELEMENT
        self.boundary = np.where(Tbound[:,-1] == self.index)[0]         # GLOBAL INDEX FOR COMPUTATIONAL DOMAIN'S BOUNDARY ELEMENTAL EDGE
        self.Nebound = len(self.boundary)                               # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.Tebound = np.zeros([self.Nebound,self.nedge], dtype=int)   # BOUNDARY EDGES CONNECTIVITY MATRIX (LOCAL INDEXES) 
        for boundedge, index in enumerate(self.boundary):
            for i in range(self.nedge):
                self.Tebound[boundedge,i] = np.where(Tbound[index,i] == self.Te)[0][0]
        return
        
        
    def ComputeStandardQuadrature2D(self,Order):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to integrations in 2D for elements which ARE NOT CUT by the interface. Hence, 
        in such elements the standard FEM integration methodology is applied (STANDARD REFERENCE SHAPE FUNCTIONS EVALUATED AT STANDARD GAUSS INTEGRATION NODES). 
        Input: - Order: Gauss quadrature order 
        
        Relevant attributes:
            ### 2D REFERENCE ELEMENT:
            #   Xig2D: GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT
            #   Wg2D: GAUSS WEIGHTS IN 2D REFERENCE ELEMENT
            #   Ng2D: NUMBER OF GAUSS INTEGRATION NODES IN 2D REFERENCE QUADRATURE
            #   N: 2D REFERENCE SHAPE FUNCTIONS EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            #   dNdxi: 2D REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            #   dNdeta: 2D REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            """
        
        # COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE IN 2D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES 
        self.XIg2D, self.Wg2D, self.Ng2D = GaussQuadrature(self.ElType,Order)
        
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        self.N, self.dNdxi, self.dNdeta = EvaluateReferenceShapeFunctions(self.XIg2D, self.ElType, self.ElOrder, self.n)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES INVERSE MATRIX 
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE ELEMENT
          
        # COMPUTE MAPPED GAUSS NODES
        self.Xg2D = self.N @ self.Xe
        # COMPUTE JACOBIAN INVERSE AND DETERMINANT
        self.invJg = np.zeros([self.Ng2D,self.dim,self.dim])
        self.detJg = np.zeros([self.Ng2D])
        Rmean = np.sum(self.Xe[:,0])/self.n   # mean elemental radial position
        for ig in range(self.Ng2D):
            self.invJg[ig,:,:], self.detJg[ig] = Jacobian(self.Xe[:,0],self.Xe[:,1],self.dNdxi[ig,:],self.dNdeta[ig,:])
            self.detJg[ig] *= 2*np.pi*Rmean   # ACCOUNT FOR AXISYMMETRICAL
            
        return    
            
    def ComputeBoundaryQuadrature(self, Order):       
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to integrations in 1D for elements which ARE NOT CUT by the interface. Hence, 
        in such elements the standard FEM integration methodology is applied (STANDARD REFERENCE SHAPE FUNCTIONS EVALUATED AT STANDARD GAUSS INTEGRATION NODES). 
        Input: - Order: Gauss quadrature order 
        
        ### 1D REFERENCE ELEMENT:
            #   Xig1D: GAUSS NODAL COORDINATES IN 1D REFERENCE ELEMENT
            #   Wg1D: GAUSS WEIGHTS IN 1D REFERENCE ELEMENT
            #   Ng1D: NUMBER OF GAUSS INTEGRATION NODES IN 1D REFERENCE QUADRATURE
            #   N1D: 1D REFERENCE SHAPE FUNCTIONS EVALUATED AT 1D REFERENCE GAUSS INTEGRATION NODES
            #   dNdxi1D: 1D REFERENCE SHAPE FUNCTIONS DERIVATIVES EVALUATED AT 1D REFERENCE GAUSS INTEGRATION NODES
        """   
         
        # COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE IN 1D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        self.XIg1D, self.Wg1D, self.Ng1D = GaussQuadrature(0,Order)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.N1D, self.dNdxi1D, foo = EvaluateReferenceShapeFunctions(self.XIg1D, 0, Order-1, self.nedge)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 1D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE 1D ELEMENT
          
        self.XIgbound = np.zeros([self.Nebound,self.Ng1D,self.dim])
        self.Nbound = np.zeros([self.Nebound,self.Ng1D,self.n])
        self.dNdxibound = np.zeros([self.Nebound,self.Ng1D,self.n])
        self.dNdetabound = np.zeros([self.Nebound,self.Ng1D,self.n])
        self.Xgbound = np.zeros([self.Nebound,self.Ng1D,self.dim])
        self.detJgbound = np.zeros([self.Nebound,self.Ng1D])
        for edge in range(self.Nebound):
            # ISOLATE EDGE NODAL COORDINATES
            Xebound = self.Xe[self.Tebound[edge,:],:]
            # IDENTIFY EDGE ON REFERENCE ELEMENT CORRESPONDING TO BOUNDARY EDGE
            XIbound = np.zeros(np.shape(Xebound))
            for i in range(len(Xebound[:,0])):
                XIbound = self.InverseMapping(Xebound[i,:])
            # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON REFERENCE BOUNDARY EDGE 
            self.XIgbound[edge,:,:] = self.N1D @ XIbound
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON REFERENCE BOUNDARY EDGE GAUSS NODES
            self.Nbound[edge,:,:], self.dNdxibound[edge,:,:], self.dNdetabound[edge,:,:] = EvaluateReferenceShapeFunctions(self.XIgbound[edge,:,:], self.ElType, self.ElOrder, self.n)
            # COMPUTE MAPPED GAUSS NODES
            self.Xgbound[edge,:,:] = self.N1D @ Xebound
            # COMPUTE JACOBIAN INVERSE AND DETERMINANT
            Rmeanbound = np.mean(Xebound[:,0])   # mean elemental radial position
            for ig in range(self.Ng1D):
                self.detJgbound[edge,ig] = Jacobian1D(Xebound[:,0],Xebound[:,1],self.dNdxi1D[ig,:])  
                self.detJgbound[edge,ig] *= 2*np.pi*Rmeanbound   # ACCOUNT FOR AXISYMMETRICAL
                
        return
    
    
    def ComputeModifiedQuadratures(self,Order):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to a 2D and 1D integration for elements which ARE CUT by the interface. 
        In this case, an adapted quadrature is computed by modifying the standard approach.  
        
        Relevant attributes:
            ### 1D REFERENCE ELEMENT:
            #   XIg1D: STANDARD GAUSS INTEGRATION NODES IN 1D REFERENCE ELEMENT
            #   Wg1D: STANDARD GAUSS INTEGRATION WEIGHTS IN 1D REFERENCE ELEMENT
            ### 2D REFERENCE ELEMENT:
            #   XIg2D: STANDARD GAUSS INTEGRATION NODES IN 2D REFERENCE ELEMENT
            #   Wg2D: STANDARD GAUSS INTEGRATION WEIGHTS IN 2D REFERENCE ELEMENT
            
            #   XIe: NODAL COORDINATES MATRIX OF 2D REFERENCE ELEMENT
            #   XIeint: INTERFACE NODES COORDINATES MATRIX IN 2D REFERENCE ELEMENT
            #   XIgint: INTERFACE GAUSS NODES COORDINATES MATRIX IN 2D REFERENCE ELEMENT, MODIFIED 1D QUADRATURE
            
            ### 2D PHYSICAL ELEMENT:
            #   Xe: NODAL COORDINATES OF 2D PHYSICAL ELEMENT 
            #   Xeint: NODAL COORDINATES OF INTERFACE IN 2D PHYSICAL ELEMENT
            #   Xgint: GAUSS NODAL COORDINATES IN 2D PHYSICAL ELEMENT 
            #   Xemod: NODAL COORDINATES OF 2D PHYSICAL ELEMENT WITH TESSELLATION
            #   Temod: CONNECTIVITY MATRIX OF 2D PHYSICAL ELEMENT WITH TESSELLATION
            
            # IN ORDER TO COMPUTE THE 2D MODIFIED QUADRATURE, WE NEED TO:
            #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
            #    2. PERFORM TESSELLATION ON THE REFERENCE ELEMENT -> OBTAIN NODAL COORDINATES OF REFERENCE SUBELEMENTS
            #    3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS 
            
            # IN ORDER TO COMPUTE THE 1D MODIFIED QUADRATURE, WE NEED TO:
            #    1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT -> OBTAIN REFERENCE INTERFACE
            #    2. MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE
        """
        
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES (2D)
        XIg2DFEM, self.Wg2D, self.Ng2D = GaussQuadrature(self.ElType,Order)
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1DFEM, self.Wg1D, self.Ng1D = GaussQuadrature(0,Order)
        
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        N, foo, foo = EvaluateReferenceShapeFunctions(XIg2DFEM, self.ElType, self.ElOrder, self.n)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1DFEM, 0, Order-1, self.nedge)
        
        #### THE STANDARD APPROACH IS MODIFIED THEN ACCORDING TO THE INTERFACE IN ORDER TO GENERATE A NEW ADAPTED QUADRATURE
        ## THE FOLLOWING METHODOLOGY IS FOLLOWED: IN ORDER TO INTERGRATE IN THE CUT ELEMENT, THE ORIGINAL PHYSICAL ELEMENT IS 
        ## TESSELLATED INTO SMALLER SUBELEMENTS. FOR EACH SUBELEMENT AN ADAPTED NUMERICAL INTEGRATION QUADRATURE IS COMPUTED FROM 
        ## THE STANDARD ONE.
        
        ######## GENERATE SUBELEMTAL STRUCTURE
        # I. PERFORM TESSELLATION ON PHYSICAL ELEMENT AND GENERATE SUBELEMENTS
        self.Temod, self.shortedge = self.Tessellation(Mode="PHYSICAL",Xemod=self.Xemod)
        self.Nsub = np.shape(self.Temod)[0]
        self.SubElements = [Element(index = subelem, ElType = self.ElType, ElOrder = self.ElOrder,
                                    Xe = self.Xemod[self.Temod[subelem,:]],
                                    Te = self.Te,
                                    LSe = None) for subelem in range(self.Nsub)]
        
        # II. DETERMINE ON TO WHICH REGION (INSIDE OR OUTSIDE) FALLS EACH SUBELEMENT
        if self.LSe[self.permu[0]] < 0:  # COMMON NODE YIELD LS < 0 -> INSIDE REGION
            Dommod = np.array([-1,1,1])
        else:
            Dommod = np.array([1,-1,-1])
            
        ######### 2D MODIFIED GAUSS NODES
        # 1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT
        XIeint = np.zeros(np.shape(self.Xeint))
        for i in range(len(self.Xeint[:,0])):
            XIeint[i,:] = self.InverseMapping(self.Xeint[i,:])
            
        # 2. DO TESSELLATION ON REFERENCE ELEMENT
        XIe = np.array([[1,0], [0,1], [0,0]])
        XIemod, TemodREF = self.Tessellation(Mode="REFERENCE",Xe=XIe,Xeint=XIeint,shortedge=self.shortedge)
        
        # 3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS AND EVALUATE INTEGRATION ENTITIES ON THEM
        for i, subelem in enumerate(self.SubElements):
            # MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS  ->> MODIFIED 2D QUADRATURE FOR SUBELEMENTS
            subelem.XIg2D = N @ XIemod[TemodREF[i,:]]
            # EVALUATE REFERENCE SHAPE FUNCTIONS ON MODIFIED REFERENCE QUADRATURE
            subelem.N, subelem.dNdxi, subelem.dNdeta = EvaluateReferenceShapeFunctions(subelem.XIg2D, subelem.ElType, subelem.ElOrder, subelem.n)
            # MAPP MODIFIED REFERENCE QUADRATURE ON PHYSICAL ELEMENT
            subelem.Xg2D = subelem.N @ self.Xe
            # ASSIGN A REGION TO EACH SUBELEMENT
            subelem.Dom = Dommod[i]
            # ASSIGN STANDARD WEIGHTS AND NUMBER OF NODES
            subelem.Ng2D = self.Ng2D
            subelem.Ng1D = self.Ng1D
            subelem.Wg2D = self.Wg2D
            subelem.Wg1D = self.Wg1D
            
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON MODIFIED QUADRATURES NODES
            subelem.invJg = np.zeros([subelem.Ng2D,subelem.dim,subelem.dim])
            subelem.detJg = np.zeros([subelem.Ng2D])
            Rmeansub = np.sum(subelem.Xe[:,0])/subelem.n   # mean subelemental radial position
            for ig in range(subelem.Ng2D):
                subelem.invJg[ig,:,:], subelem.detJg[ig] = Jacobian(subelem.Xe[:,0],subelem.Xe[:,1],subelem.dNdxi[ig,:],subelem.dNdeta[ig,:])
                subelem.detJg[ig] *= 2*np.pi*Rmeansub   # ACCOUNT FOR AXISYMMETRICAL
                
                
        ######### 1D MODIFIED GAUSS NODES
        # 2. MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE ->> MODIFIED 1D QUADRATURE FOR INTERFACE
        self.XIgint = N1D @ XIeint
        
        # EVALUATE 2D REFERENCE SHAPE FUNCTION ON INTERFACE MODIFIED QUADRATURE
        self.Nint, self.dNdxiint, self.dNdetaint = EvaluateReferenceShapeFunctions(self.XIgint, self.ElType, self.ElOrder, self.n)
        # MAPP REFERENCE INTERFACE MODIFIED QUADRATURE ON PHYSICAL ELEMENT 
        self.Xgint = N1D @ self.Xeint
        
        self.detJgint = np.zeros([self.Ng1D])
        Rmeanint = np.mean(self.Xeint[:,0])   # mean interface radial position
        for ig in range(self.Ng1D):
            self.detJgint[ig] = Jacobian1D(self.Xeint[:,0],self.Xeint[:,1],dNdxi1D[ig,:])  
            self.detJgint[ig] *= 2*np.pi*Rmeanint   # ACCOUNT FOR AXISYMMETRICAL
            
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
            self.NormalVec = -1*ntest
        return 
    
    def BoundaryNormal(self,Xmax,Xmin,Ymax,Ymin):
        """ This function computes the boundary edge(s) normal vector(s) pointing outwards. """
        
        self.NormalVec = np.zeros([self.Nebound,self.dim])
        for edge in range(self.Nebound):
            dx = self.Xe[self.Tebound[edge,1],0] - self.Xe[self.Tebound[edge,0],0]
            dy = self.Xe[self.Tebound[edge,1],1] - self.Xe[self.Tebound[edge,0],1]
            ntest = np.array([-dy, dx])   # test this normal vector
            ntest = ntest/np.linalg.norm(ntest)   # normalize
            Xintmean = np.array([np.mean(self.Xe[self.Tebound[edge,:],0]), np.mean(self.Xe[self.Tebound[edge,:],1])])  # mean point on interface
            Xtest = Xintmean + 2*ntest  # physical point on which to test if outside of computational domain 
            
            # CHECK IF TEST POINT IS OUTSIDE COMPUTATIONAL DOMAIN
            if Xtest[0] < Xmin or Xmax < Xtest[0] or Xtest[1] < Ymin or Ymax < Xtest[1]:  
                self.NormalVec[edge,:] = ntest
            else: 
                self.NormalVec[edge,:] = -1*ntest
        
        return
    
    
    def IntegrateElementalDomainTerms(self,SourceTermg,LHS,RHS):
        """ Input: - SourceTermg: source term (plasma current) evaluated at physical gauss integration nodes
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector
                    """
        
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.Ng2D):  
            # SHAPE FUNCTIONS GRADIENT
            Ngrad = np.array([self.dNdxi[ig,:],self.dNdeta[ig,:]])
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
            for i in range(self.n):   # ROWS ELEMENTAL MATRIX
                for j in range(self.n):   # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) *(Jacobiano*2pi*rad) ]  
                    LHS[self.Te[i],self.Te[j]] -= np.transpose((self.invJg[ig,:,:]@Ngrad[:,i]))@(self.invJg[ig,:,:]@Ngrad[:,j])*self.detJg[ig]*self.Wg2D[ig]
                    ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j *(Jacobiano*2pi*rad) ]  ONLY RESPECT TO R
                    LHS[self.Te[i],self.Te[j]] -= (1/self.Xg2D[ig,0])*self.N[ig,j] * (self.invJg[ig,0,:]@Ngrad[:,i])*self.detJg[ig]*self.Wg2D[ig]
                # COMPUTE RHS VECTOR TERMS [ (source term)*N_i*(Jacobiano *2pi*rad) ]
                RHS[self.Te[i]] += SourceTermg[ig] * self.N[ig,i] *self.detJg[ig]*self.Wg2D[ig]
        return 
    
    
    def IntegrateElementalInterfaceTerms(self,PHI_Dg,beta,LHS,RHS):
        """ Input: - PHI_Dg: Interface condition, evaluated at physical gauss integration nodes
                   - beta: Nitsche's method penalty parameter
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector 
                    """
    
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.Ng1D):  
            # SHAPE FUNCTIONS GRADIENT
            Ngrad = np.array([self.dNdxiint[ig,:],self.dNdetaint[ig,:]])
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
            for i in range(self.n):  # ROWS ELEMENTAL MATRIX
                for j in range(self.n):  # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) *(Jacobiano*2pi*rad) ]  
                    LHS[self.Te[i],self.Te[j]] += self.Nint[ig,i] * self.NormalVec @ Ngrad[:,j] * self.detJgint[ig] * self.Wg1D[ig]
                    ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) *(Jacobiano*2pi*rad) ]
                    LHS[self.Te[i],self.Te[j]] += self.NormalVec @ Ngrad[:,i]*(self.Nint[ig,j] * self.detJgint[ig] * self.Wg1D[ig])
                    ### PENALTY TERM   [ beta * (N_i*N_j) *(Jacobiano*2pi*rad) ]
                    LHS[self.Te[i],self.Te[j]] += beta * self.Nint[ig,i] * self.Nint[ig,j] * self.detJgint[ig] * self.Wg1D[ig]
                # COMPUTE RHS VECTOR TERMS 
                ### SYMMETRIC NITSCHE'S METHOD TERM  [ PHI_D * (n dot nabla(N_i)) * (Jacobiano *2pi*rad) ]
                RHS[self.Te[i]] +=  PHI_Dg[ig] * self.NormalVec @ Ngrad[:,i] * self.detJgint[ig] * self.Wg1D[ig]
                ### PENALTY TERM   [ beta * N_i * PHI_D *(Jacobiano*2pi*rad) ]
                RHS[self.Te[i]] +=  beta * PHI_Dg[ig] * self.Nint[ig,i] * self.detJgint[ig] * self.Wg1D[ig]
        return 
    
    
    def IntegrateElementalBoundaryTerms(self,PHI_Bg,beta,LHS,RHS):
        """ Input: - PHI_Bg: Boundary condition, evaluated at physical gauss integration nodes
                   - beta: Nitsche's method penalty parameter
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector 
                    """

        # LOOP OVER EDGES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        for edge in range(self.Nebound):
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(self.Ng1D):  
                # SHAPE FUNCTIONS GRADIENT
                Ngrad = np.array([self.dNdxibound[edge,ig,:],self.dNdetabound[edge,ig,:]])
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                for i in range(self.n):  # ROWS ELEMENTAL MATRIX
                    for j in range(self.n):  # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) *(Jacobiano*2pi*rad) ]  
                        LHS[self.Te[i],self.Te[j]] += self.Nbound[edge,ig,i] * self.NormalVec[edge,:] @ Ngrad[:,j] * self.detJgbound[edge,ig] * self.Wg1D[ig]
                        ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) *(Jacobiano*2pi*rad) ]
                        LHS[self.Te[i],self.Te[j]] += self.NormalVec[edge,:] @ Ngrad[:,i]*(self.Nbound[edge,ig,j] * self.detJgbound[edge,ig] * self.Wg1D[ig])
                        ### PENALTY TERM   [ beta * (N_i*N_j) *(Jacobiano*2pi*rad) ]
                        LHS[self.Te[i],self.Te[j]] += beta * self.Nbound[edge,ig,i] * self.Nbound[edge,ig,j] * self.detJgbound[edge,ig] * self.Wg1D[ig]
                    # COMPUTE RHS VECTOR TERMS 
                    ### SYMMETRIC NITSCHE'S METHOD TERM  [ PHI_D * (n dot nabla(N_i)) * (Jacobiano *2pi*rad) ]
                    RHS[self.Te[i]] +=  PHI_Bg[edge,ig] * self.NormalVec[edge,:] @ Ngrad[:,i] * self.detJgbound[edge,ig] * self.Wg1D[ig]
                    ### PENALTY TERM   [ beta * N_i * PHI_D *(Jacobiano*2pi*rad) ]
                    RHS[self.Te[i]] +=  beta * PHI_Bg[edge,ig] * self.Nbound[edge,ig,i] * self.detJgbound[edge,ig] * self.Wg1D[ig]
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
    nedge = elemOrder + 1
    return n, nedge
    