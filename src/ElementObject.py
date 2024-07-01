""" THIS SCRIPT DEFINES THE ATTRIBUTES AND METHODS CORRESPONDING TO OBJECT ELEMENT. THIS CLASS SHALL GATHER ALL THE RELEVANT INFORMATION (COORDINATES,
QUADRATURES, NODAL VALUES...) FOR A SINGLE ELEMENT IN THE MESH. """

from src.GaussQuadrature import *
from src.ShapeFunctions import *
from scipy import optimize
from src.SegmentObject import *

class Element:
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,PlasmaLSe,VacVessLSe):
        
        self.index = index                                              # GLOBAL INDEX ON COMPUTATIONAL MESH
        self.ElType = ElType                                            # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
        self.ElOrder = ElOrder                                          # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.numedges = ElementalNumberOfEdges(ElType)                  # ELEMENTAL NUMBER OF EDGES
        self.n, self.nedge = ElementalNumberOfNodes(ElType, ElOrder)    # NUMBER OF NODES PER ELEMENT, PER ELEMENTAL EDGE
        if self.ElType == 1:
            self.numvertices = 3
        elif self.ElType == 2:
            self.numvertices = 4
        self.Xe = Xe                                                    # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.dim = len(Xe[0,:])                                         # SPATIAL DIMENSION
        self.Te = Te                                                    # ELEMENTAL CONNECTIVITIES
        self.PlasmaLSe = PlasmaLSe                                      # ELEMENTAL NODAL PLASMA REGION LEVEL-SET VALUES
        self.VacVessLSe = VacVessLSe                                    # ELEMENTAL NODAL VACUUM VESSEL FIRST WALL LEVEL-SET VALUES
        self.PHIe = np.zeros([self.n])                                  # ELEMENTAL NODAL PHI VALUES
        self.Dom = None                                                 # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; 0: "PLASMA INTERFACE"; +1: "VACUUM" ; +2: FIRST WALL ; +3: "EXTERIOR")
        
        # INTEGRATION QUADRATURES ENTITIES
        self.Ng2D = None            # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 2D GAUSS QUADRATURE
        self.XIg2D = None           # STANDARD 2D GAUSS INTEGRATION NODES 
        self.Wg2D = None            # STANDARD 2D GAUSS INTEGRATION WEIGTHS
        self.N = None               # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES 
        self.dNdxi = None           # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.dNdeta = None          # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT STANDARD 2D GAUSS INTEGRATION NODES
        self.Xg2D = None            # PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM 2D REFERENCE ELEMENT
        self.invJg = None           # INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES 
        
        ### ATTRIBUTES FOR INTERFACE ELEMENTS
        self.Neint = None           # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.InterEdges = None      # ARRAY CONTAINING THE ELEMENTAL EDGES/CUTS CORRESPONDING TO INTERFACES
        self.Nsub = None            # NUMBER OF SUBELEMENTS GENERATED IN TESSELLATION
        self.XeTESS = None          # MODIFIED ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.TeTESS = None          # MODIFIED CONNECTIVITY MATRIX FOR TESSELLED INTERFACE PHYSICAL ELEMENT
        self.DomTESS = None         # DOMAIN WHERE THE TESSELLED ELEMENT SUBELEMENTS LIE (-1: "PLASMA"; +1: "VACUUM" ; +3: "EXTERIOR")
        self.XIeTESS = None         # MODIFIED REFERENCE ELEMENT NODAL MATRIX (PHYSICAL COORDINATES)
        self.TeTESSREF = None       # MODIFIED CONNECTIVITY MATRIX FOR TESSELLED INTERFACE REFERENCE ELEMENT
        self.XIeint = None
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
        Xi = self.InverseMapping(X)
        for i in range(self.n):
            N, foo, foo = ShapeFunctionsReference(Xi, self.ElType, self.ElOrder, i+1)
            F += N*Fe[i]
        return F
    
    
    ##################################################################################################
    ################################ CUTELEMENTS INTERFACE ###########################################
    ##################################################################################################
    
    
    def InterfaceLinearApproximation(self,interface_index):
        """ FUNCTION COMPÙTING THE INTERSECTION POINTS BETWEEN THE ELEMENT EDGES AND THE INTERFACE CUTTING THE ELEMENT (PLASMA/VACUUM INTERFACE OR VACUUM VESSEL FIRST WALL).  
            THE CUTTING SEGMENT IS STORED AS A SEGMENT OBJECT (DEFINED IN FILE SegmentObject.py) IN ELEMENT ATTRIBUTE InterEdges
            
            COMPUTED ATTRIBUTES:
                * FOR ELEMENT
                    - Neint: NUMBER OF INTERFACES IN ELEMENT
                * FOR INTERFACE SEGMENT
                    - Xeint: PHYSICAL INTERFACE SEGMENT VERTICES COORDINATES 
                    - inter_edges: LOCAL INDICES OF VERTICES WHERE SEGMENT ENDS ARE LOCATED
            """
        
        # READ LEVEL-SET NODAL VALUES
        if self.Dom == 0:  # PLASMA/VACUUM INTERFACE ELEMENT
            LSe = self.PlasmaLSe  
        if self.Dom == 2:  # VACUUM VESSEL FIRST WALL ELEMENT
            LSe = self.VacVessLSe
        
        # OBTAIN INTERSECTION COORDINATES FOR EACH EDGE:
        self.Neint = 1
        self.InterEdges = [Segment(index = interface_index,
                                    ElOrder = 1,   # WE ARE ONLY INTEREDTED IN THE VERTICES OF THE SEGMENT CUTING THE ELEMENT ->> 2 NODES 
                                    Xeint = np.zeros([2,self.dim]),  # HERE WILL BE STORED THE SEGMENT VERTICES COORDINATES
                                    int_edges = np.zeros([2,2])) for interf in range(self.Neint)]   # HERE WILL BE STORED THE LOCAL INDICES OF THE VERTICES DEFINING THE EDGES WHERE THE INTERSECTION POINTS LIE
        
        for interf in range(self.Neint):
            k = 0
            self.InterEdges[interf].int_edges = np.zeros([2,2],dtype=int)
            for i in range(self.numedges):  # Loop over elemental edges
                # Check for sign change along the edge
                if LSe[i] * LSe[(i + 1) % self.numedges] < 0:
                    # Store on which edge is located the intersection point
                    self.InterEdges[interf].int_edges[k,:] = [i,(i + 1) % self.numedges]
                    # Interpolate to find intersection point
                    t = LSe[i] / (LSe[i] - LSe[(i + 1) % self.numedges])
                    self.InterEdges[interf].Xeint[k,:] = (1 - t) * self.Xe[i,:] + t * self.Xe[(i + 1) % self.numedges,:]
                    k += 1
        
        return 
    
    
    def ComputationalDomainBoundaryEdges(self,Tbound):
        """ This function finds for each element the edges lying on the computational domain's boundary. The different elemental attributes are set-up accordingly.
        
        Input: - Tbound: # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        
        COMPUTED ATTRIBUTES:
                * FOR ELEMENT
                    - Neint: NUMBER OF INTERFACES IN ELEMENT
                * FOR INTERFACE SEGMENT
                    - Xeint: PHYSICAL INTERFACE SEGMENT VERTICES COORDINATES 
                    - inter_edges: LOCAL INDICES OF VERTICES WHERE SEGMENT ENDS ARE LOCATED
                """
        
        # LOOK WHICH BOUNDARIES ARE ASSOCIATED TO THE ELEMENT
        interface = np.where(Tbound[:,-1] == self.index)[0]         # GLOBAL INDEX FOR COMPUTATIONAL DOMAIN'S BOUNDARY ELEMENTAL EDGE
        self.Neint = len(interface)                                 # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        
        self.InterEdges = [Segment(index = interface[interf],
                                    ElOrder = 1,   # WE ARE ONLY INTEREDTED IN THE VERTICES OF THE SEGMENT CUTING THE ELEMENT ->> 2 NODES 
                                    Xeint = np.zeros([2,self.dim]),   # HERE WILL BE STORED THE SEGMENT VERTICES COORDINATES
                                    int_edges = np.zeros([2])) for interf in range(self.Neint)]   # HERE WILL BE STORED THE LOCAL INDICES OF THE VERTICES DEFINING THE EDGES WHERE THE INTERSECTION POINTS LIE
        
        for interf, index in enumerate(interface):
            # FIND LOCAL INDEXES OF NODES ON EDGE 
            self.InterEdges[interf].int_edges = np.zeros([len(Tbound[0,:-1])], dtype=int)
            for i in range(len(Tbound[0,:-1])):
                self.InterEdges[interf].int_edges[i] = np.where(Tbound[index,i] == self.Te)[0][0]
            # COORDINATES OF NODES ON EDGE
            self.InterEdges[interf].Xeint = self.Xe[self.InterEdges[interf].int_edges,:]
            
        return 
    
    
    ##################################################################################################
    ##################################### INTERFACE NORMALS ##########################################
    ##################################################################################################
    
    def InterfaceNormal(self):
        """ This function computes the interface normal vector pointing outwards. """
        
        for edge in range(self.Neint):
            dx = self.InterEdges[edge].Xeint[1,0] - self.InterEdges[edge].Xeint[0,0]
            dy = self.InterEdges[edge].Xeint[1,1] - self.InterEdges[edge].Xeint[0,1]
            ntest = np.array([-dy, dx])   # test this normal vector
            ntest = ntest/np.linalg.norm(ntest)   # normalize
            Xintmean = np.array([np.mean(self.InterEdges[edge].Xeint[:,0]), np.mean(self.InterEdges[edge].Xeint[:,1])])  # mean point on interface
            Xtest = Xintmean + 3*ntest  # physical point on which to test the Level-Set 
            
            # INTERPOLATE LEVEL-SET ON XTEST
            LStest = 0
            # LEVEL-SET NODAL VALUES
            if self.Dom == 0:  # ELEMENT CONTAINING PLASMA/VACUUM INTERFACE
                LSe = self.PlasmaLSe
            elif self.Dom == 2:  # ELEMENT CONTAINING VACUUM VESSEL FIRST WALL
                LSe = self.VacVessLSe
                
            # INTERPOLATE LEVEL-SET USING WEIGHTS INVERSELY PROPORTIONAL TO DISTANCE
            for inode in range(self.n):
                # COMPUTE INTERPOLATION WEIGHTS
                w = 1.0/np.sqrt((Xtest[0] - self.Xe[inode,0])**2 + (Xtest[1] - self.Xe[inode,1])**2)
                # INTERPOLATE LEVEL-SET
                LStest += w*LSe[inode]
                
            # CHECK SIGN OF LEVEL-SET 
            if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
                self.InterEdges[edge].NormalVec = ntest
            else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
                self.InterEdges[edge].NormalVec = -1*ntest
        return 
    
    def ComputationalDomainBoundaryNormal(self,Xmax,Xmin,Ymax,Ymin):
        """ This function computes the boundary edge(s) normal vector(s) pointing outwards. """
        
        for edge in range(self.Neint):
            dx = self.InterEdges[edge].Xeint[1,0] - self.InterEdges[edge].Xeint[0,0]
            dy = self.InterEdges[edge].Xeint[1,1] - self.InterEdges[edge].Xeint[0,1]
            ntest = np.array([-dy, dx])   # test this normal vector
            ntest = ntest/np.linalg.norm(ntest)   # normalize
            Xintmean = np.array([np.mean(self.InterEdges[edge].Xeint[:,0]), np.mean(self.InterEdges[edge].Xeint[:,1])])  # mean point on interface
            Xtest = Xintmean + 3*ntest  # physical point on which to test if outside of computational domain 
            
            # CHECK IF TEST POINT IS OUTSIDE COMPUTATIONAL DOMAIN
            if Xtest[0] < Xmin or Xmax < Xtest[0] or Xtest[1] < Ymin or Ymax < Xtest[1]:  
                self.InterEdges[edge].NormalVec = ntest
            else: 
                self.InterEdges[edge].NormalVec = -1*ntest
        return
    
    
    ##################################################################################################
    ################################ ELEMENTAL TESSELLATION ##########################################
    ##################################################################################################
    
    @staticmethod
    def CheckNodeOnEdge(x,numvertices,Xe,TOL):
        """ Function which checks if point with coordinates x is on any edge of the element with nodal coordinates Xe. """
        edgecheck = False
        for vertexnode in range(numvertices):
            i = vertexnode
            j = (vertexnode+1)%numvertices
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
        
    
    def ElementTessellation(self):
        """ This function performs the TESSELLATION of an element with nodal coordinates Xe and interface coordinates Xeint (intersection with edges) 
        
        Output: - TeTESS: Tessellation connectivity matrix such that 
                        TeTESS = [[Connectivities for subelement 1]
                                  [Connectivities for subelement 2]
                                                ...          
                - XeTESS: Nodal coordinates matrix storing the coordinates of the element vertex and interface points                   
                """
                
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        # HOWEVER, WHEN LOOKING FOR THE LINEAR APPROXIMATION OF THE PHYSICAL INTERFACE THIS PROCESS IS ALREADY DONE, THEREFORE WE CAN SKIP IT. 
        # IF INPUT Xemod IS PROVIDED, THE TESSELLATION IS DONE ACCORDINGLY TO MODIFIED NODAL MATRIX Xemod WHICH IS ASSUMED TO HAS THE PREVIOUSLY DESCRIBED STRUCTURE.
        # IF NOT, THE COMMON NODE IS DETERMINED (THIS IS THE CASE FOR INSTANCE WHEN THE REFERENCE ELEMENT IS TESSELLATED).
                
        # LEVEL-SET NODAL VALUES
        if self.Dom == 0:  # ELEMENT CONTAINING PLASMA/VACUUM INTERFACE
            LSe = self.PlasmaLSe
        elif self.Dom == 2:  # ELEMENT CONTAINING VACUUM VESSEL FIRST WALL
            LSe = self.VacVessLSe
                
        for interf in range(self.Neint):  # IN ALL CASES THIS IS ALWAYS GONNA BE 1 ITERATION, BECAUSE FOR CUT ELEMENTS THE INTERFACE IS APPROXIMATED USING A SINGLE SEGMENT
            if self.ElType == 1:  # TRIANGULAR ELEMENT
                # MODIFIED NODAL MATRIX AND CONECTIVITIES, ACCOUNTING FOR 3 SUBTRIANGLES 
                XeTESS = np.concatenate((self.Xe[:self.numvertices,:],self.InterEdges[interf].Xeint),axis=0)
                TeTESS = np.zeros([3,3],dtype=int)
                
                # LOOK FOR COMMON NODE BY USING SIGN OF prod(LSe)
                #  -> IF prod(LSe) > 0, THEN COMMON NODE HAS LS > 0
                #  -> IF prod(LSe) < 0, THEN COMMON NODE HAS LS < 0
                for ivertex in range(self.numvertices):
                    if np.sign(LSe[ivertex]) == np.sign(np.prod(LSe[:self.numvertices])):
                        commonnode = ivertex
                        break
                edgenode = np.zeros([2],dtype=int)
                distance = np.zeros([2])
                for iedge in range(2):
                    for node in self.InterEdges[interf].int_edges[iedge,:]:
                        if node != commonnode:
                            edgenode[iedge] = node
                            distance[iedge] = np.linalg.norm(self.Xe[node,:]-self.InterEdges[interf].Xeint[iedge,:])
                
                TeTESS[0,:] = [commonnode, 3,4]  # SUBELEMEMENT WITH COMMON NODE
                if distance[0] < distance[1]:
                    TeTESS[1,:] = [edgenode[0],edgenode[1],3]
                    TeTESS[2,:] = [edgenode[1],3,4]
                if distance[0] >= distance[1]:
                    TeTESS[1,:] = [edgenode[1],edgenode[0],4]
                    TeTESS[2,:] = [edgenode[0],3,4]
                    
                if self.Dom == 0:   # PLASMA/VACUUM INTERFACE
                    if LSe[commonnode] < 0:
                        DomTESS = np.array([-1,1,1])
                    else:
                        DomTESS = np.array([1,-1,-1])
                elif self.Dom == 2:  # VACUUM VESSEL FIRST WALL INTERFACE
                    if LSe[commonnode] < 0:
                        DomTESS = np.array([1,3,3])
                    else:
                        DomTESS = np.array([3,1,1])
                
            elif self.ElType == 2:  # QUADRILATERAL ELEMENT
                # MODIFIED NODAL MATRIX
                XeTESS = np.concatenate((self.Xe[:self.numvertices,:],self.InterEdges[interf].Xeint),axis=0)
                # LOOK FOR TESSELLATION CONFIGURATION BY USING SIGN OF prod(LSe)
                #  -> IF prod(LSe) > 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO 2 CHILD QUADRILATERAL ELEMENTS LS > 0
                #  -> IF prod(LSe) < 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO PENTAGON AND TRIANGLE
                if np.prod(LSe[:self.numvertices]) > 0:
                    # MODIFIED CONECTIVITIES, ACCOUNTING FOR 2 SUBQUADRILATERALS
                    TeTESS = np.zeros([2,4],dtype=int)
                    nodepositive = []
                    nodenegative = []
                    for iedge, edge in enumerate(self.InterEdges[interf].int_edges):
                        for node in edge:
                            if LSe[node] > 0:
                                nodepositive.append(node)
                            else:
                                nodenegative.append(node)
                    TeTESS[0,:] = np.concatenate((np.array([4]),np.array(nodepositive),np.array([5])),axis=0)
                    TeTESS[1,:] = np.concatenate((np.array([4]),np.array(nodenegative),np.array([5])),axis=0)
                    
                    if self.Dom == 0:   # PLASMA/VACUUM INTERFACE
                        DomTESS = np.array([1,-1])
                    elif self.Dom == 2:  # VACUUM VESSEL FIRST WALL INTERFACE
                        DomTESS = np.array([3,1])
                    
                else:
                    # MODIFIED CONECTIVITIES, ACCOUNTING FOR 4 SUBTRIANGLES
                    TeTESS = np.zeros([4,3],dtype=int)
                    # LOOK FOR DIFFERENT LS SIGN VERTEX NODE
                    if len(np.where(LSe[:self.numvertices]<0)[0]) > len(np.where(LSe[:self.numvertices]>0)[0]):  # if more LSe < 0 than LSe > 0 -> DIFFERENT LS SIGN VERTEX IS LS > 0
                        for inode in range(self.numvertices):
                            if LSe[inode] > 0:
                                commonnode = inode
                            if np.isin(inode,np.concatenate((self.InterEdges[interf].int_edges[0],self.InterEdges[interf].int_edges[1]),axis=0)):
                                pass
                            else:
                                oppositenode = inode
                    else:  # if less LSe < 0 than LSe > 0 -> DIFFERENT LS SIGN VERTEX IS LS < 0
                        for inode in range(self.numvertices):
                            if LSe[inode] < 0:
                                commonnode = inode
                            if np.isin(inode,np.concatenate((self.InterEdges[interf].int_edges[0],self.InterEdges[interf].int_edges[1]),axis=0)):
                                pass
                            else:
                                oppositenode = inode
                    edgenode = np.zeros([2],dtype=int)
                    for iedge, edge in enumerate(self.InterEdges[interf].int_edges):
                        for node in edge:
                            if node != commonnode:
                                edgenode[iedge] = node
                        
                    TeTESS[0,:] = [commonnode,4,5]
                    TeTESS[1,:] = [4,edgenode[0],oppositenode]
                    TeTESS[2,:] = [5,edgenode[1],oppositenode]
                    TeTESS[3,:] = [4,5,oppositenode]
                    
                    if self.Dom == 0:   # PLASMA/VACUUM INTERFACE
                        if LSe[commonnode] < 0:
                            DomTESS = np.array([-1,1,1,1])
                        else:
                            DomTESS = np.array([1,-1,-1,-1])
                    elif self.Dom == 2:  # VACUUM VESSEL FIRST WALL INTERFACE
                        if LSe[commonnode] < 0:
                            DomTESS = np.array([1,3,3,3])
                        else:
                            DomTESS = np.array([3,1,1,1])
                
            return XeTESS, TeTESS, DomTESS
        
        
    def ReferenceElementTessellation(self,XIeint):
        """ This function performs the TESSELLATION of an element with nodal coordinates Xe and interface coordinates Xeint (intersection with edges) 
        
        Output: - TeTESS: Tessellation connectivity matrix such that 
                        TeTESS = [[Connectivities for subelement 1]
                                  [Connectivities for subelement 2]
                                                ...          
                - XeTESS: Nodal coordinates matrix storing the coordinates of the element vertex and interface points                   
                """
                
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        # HOWEVER, WHEN LOOKING FOR THE LINEAR APPROXIMATION OF THE PHYSICAL INTERFACE THIS PROCESS IS ALREADY DONE, THEREFORE WE CAN SKIP IT. 
        # IF INPUT Xemod IS PROVIDED, THE TESSELLATION IS DONE ACCORDINGLY TO MODIFIED NODAL MATRIX Xemod WHICH IS ASSUMED TO HAS THE PREVIOUSLY DESCRIBED STRUCTURE.
        # IF NOT, THE COMMON NODE IS DETERMINED (THIS IS THE CASE FOR INSTANCE WHEN THE REFERENCE ELEMENT IS TESSELLATED).
    
        if self.ElType == 1:  # TRIANGULAR ELEMENT
            
            XIe = np.array([[1,0], [0,1], [0,0]])
            
            # MODIFIED NODAL MATRIX AND CONECTIVITIES, ACCOUNTING FOR 3 SUBTRIANGLES 
            XeTESS = np.concatenate((XIe,XIeint),axis=0)
            TeTESS = np.zeros([3,3],dtype=int)
            
            # LOOK FOR COMMON NODE
            edgenodes = np.zeros([2,2],dtype=int)
            edgenode = np.zeros([2],dtype=int)
            distance = np.zeros([2])
            
            for i in range(2):
                edgenodes[i,:] = self.CheckNodeOnEdge(XIeint[i,:],self.numvertices,XIe,1e-4)
            commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop()
            # LOOK FOR NODE ON EDGE WHERE INTERSECTION POINT LIES BUT OTHER THAN COMMON NODE AND COMPUTE DISTANCE
            for i in range(2):
                edgenodeset = set(edgenodes[i,:])
                edgenodeset.remove(commonnode)
                edgenode[i] = edgenodeset.pop()
                distance[i] = np.linalg.norm(XIeint[i,:]-XIe[edgenode[i],:])
                        
            #print("common node = ", commonnode)
            #print("edge nodes = ", edgenode)
            #print("distances = ", distance)
            
            TeTESS[0,:] = [commonnode, 3,4]  # SUBELEMEMENT WITH COMMON NODE
            if distance[0] < distance[1]:
                TeTESS[1,:] = [edgenode[0],edgenode[1],3]
                TeTESS[2,:] = [edgenode[1],3,4]
            if distance[0] >= distance[1]:
                TeTESS[1,:] = [edgenode[1],edgenode[0],4]
                TeTESS[2,:] = [edgenode[0],3,4]
            
        elif self.ElType == 2:  # QUADRILATERAL ELEMENT
            
            XIe = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
            
            # MODIFIED NODAL MATRIX
            XeTESS = np.concatenate((XIe,XIeint),axis=0)
            
            # DIFFERENT CASES DEPENDING ON PHYSICAL ELEMENT TESSELLATION
            if self.Nsub == 2:
                # MODIFIED CONECTIVITIES, ACCOUNTING FOR 2 SUBQUADRILATERALS
                TeTESS = np.array([[4,0,1,5],
                                    [4,3,3,5]])
                
            elif self.Nsub == 4:
                # MODIFIED CONECTIVITIES, ACCOUNTING FOR 4 SUBTRIANGLES
                TeTESS = np.zeros([4,3],dtype=int)
                
                # LOOK FOR COMMON NODE
                edgenodes = np.zeros([2,2],dtype=int)
                edgenode = np.zeros([2],dtype=int)
                distance = np.zeros([2])
                for i in range(2):
                    edgenodes[i,:] = self.CheckNodeOnEdge(XIeint[i,:],self.numvertices,XIe,1e-4)
                commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop()
                # LOOK FOR NODE ON EDGE WHERE INTERSECTION POINT LIES BUT OTHER THAN COMMON NODE
                for i in range(2):
                    edgenodeset = set(edgenodes[i,:])
                    edgenodeset.remove(commonnode)
                    edgenode[i] = edgenodeset.pop()
                # LOOK FOR OPPOSITE NODE
                for i in range(4):  # LOOP OVER VERTEX
                    if np.isin(edgenodes, i).any():  # CHECK IF VERTEX IS PART OF THE EDGES ON WHICH THE INTERSECTION POINTS LIE
                        pass
                    else:
                        oppositenode = i
                        
                TeTESS[0,:] = [commonnode,4,5]
                TeTESS[1,:] = [4,edgenode[0],oppositenode]
                TeTESS[2,:] = [5,edgenode[1],oppositenode]
                TeTESS[3,:] = [4,5,oppositenode]
                
        return XeTESS, TeTESS
        
        
    ##################################################################################################
    ############################### ELEMENTAL NUMERICAL QUADRATURES ##################################
    ##################################################################################################
        
    def ComputeStandardQuadrature2D(self,Order):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to integrations in 2D for elements which ARE NOT CUT BY ANY INTERFACE. Hence, 
        in such elements the standard FEM integration methodology is applied (STANDARD REFERENCE SHAPE FUNCTIONS EVALUATED AT STANDARD GAUSS INTEGRATION NODES). 
        Input: - Order: Gauss quadrature order 
        
        Relevant attributes:
            ### 2D REFERENCE ELEMENT:
            #   XIg2D: GAUSS NODAL COORDINATES IN 2D REFERENCE ELEMENT
            #   Wg2D: GAUSS WEIGHTS IN 2D REFERENCE ELEMENT
            #   Ng2D: NUMBER OF GAUSS INTEGRATION NODES IN 2D REFERENCE QUADRATURE
            #   N: 2D REFERENCE SHAPE FUNCTIONS EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            #   dNdxi: 2D REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            #   dNdeta: 2D REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT 2D REFERENCE GAUSS INTEGRATION NODES
            #   Xg2D: PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM 2D REFERENCE ELEMENT
            #   detJg: INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
            #   invJg: MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
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
            
    def ComputeComputationalDomainBoundaryQuadrature(self, Order):       
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
        XIg1D, Wg1D, Ng1D = GaussQuadrature(0,Order)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1D, 0, Order-1, self.nedge)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 1D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE 1D ELEMENT
        
        for edge in range(self.Neint):
            self.InterEdges[edge].Ngaussint = Ng1D
            self.InterEdges[edge].Wgint = Wg1D
            self.InterEdges[edge].detJgint = np.zeros([self.InterEdges[edge].Ngaussint])
            
            # IDENTIFY EDGE ON REFERENCE ELEMENT CORRESPONDING TO VACUUM VESSEL FIRST WALL EDGE
            XIeint = np.zeros(np.shape(self.InterEdges[edge].Xeint[:,:]))
            for i in range(2):
                XIeint[i,:] = self.InverseMapping(self.InterEdges[edge].Xeint[i,:])
            # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON REFERENCE VACUUM VESSEL FIRST WALL EDGE 
            self.InterEdges[edge].XIgint = N1D @ XIeint
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON REFERENCE VACUUM VESSEL FIRST WALL EDGE GAUSS NODES
            self.InterEdges[edge].Nint, self.InterEdges[edge].dNdxiint, self.InterEdges[edge].dNdetaint = EvaluateReferenceShapeFunctions(self.InterEdges[edge].XIgint, self.ElType, self.ElOrder, self.n)
            # COMPUTE MAPPED GAUSS NODES
            self.InterEdges[edge].Xgint = N1D @ self.InterEdges[edge].Xeint
            # COMPUTE JACOBIAN INVERSE AND DETERMINANT
            Rmeanint = np.mean(self.InterEdges[edge].Xeint[:,0])   # mean elemental radial position
            for ig in range(self.InterEdges[edge].Ngaussint):
                self.InterEdges[edge].detJgint[ig] = Jacobian1D(self.InterEdges[edge].Xeint[:,0],self.InterEdges[edge].Xeint[:,1],dNdxi1D[ig])  
                self.InterEdges[edge].detJgint[ig] *= 2*np.pi*Rmeanint   # ACCOUNT FOR AXISYMMETRICAL
                
        return
    
    
    def ComputeModifiedQuadratures(self,Order):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to a 2D and 1D integration for elements which ARE CUT BY AN INTERFACE. 
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
        
        #### THE STANDARD APPROACH IS MODIFIED THEN ACCORDING TO THE INTERFACE IN ORDER TO GENERATE A NEW ADAPTED QUADRATURE
        ## THE FOLLOWING METHODOLOGY IS FOLLOWED: IN ORDER TO INTERGRATE IN THE CUT ELEMENT, THE ORIGINAL PHYSICAL ELEMENT IS 
        ## TESSELLATED INTO SMALLER SUBELEMENTS. FOR EACH SUBELEMENT AN ADAPTED NUMERICAL INTEGRATION QUADRATURE IS COMPUTED FROM 
        ## THE STANDARD ONE.
        
        ######## GENERATE SUBELEMENTAL STRUCTURE
        # PERFORM TESSELLATION ON PHYSICAL ELEMENT AND GENERATE SUBELEMENTS
        self.XeTESS, self.TeTESS, self.DomTESS = self.ElementTessellation()
        self.Nsub = np.shape(self.TeTESS)[0]
        self.SubElements = [None]*self.Nsub
        for subelem in range(self.Nsub):
            if len(self.XeTESS[self.TeTESS[subelem,:],0]) == 3:  # LINEAR TRIANGULAR SUBELEM
                SubElType = 1
            elif len(self.XeTESS[self.TeTESS[subelem,:],0]) == 4:  # LINEAR QUADRILATERAL SUBELEM
                SubElType = 2
            self.SubElements[subelem] = Element(index = subelem, ElType = SubElType, ElOrder = 1,
                                    Xe = self.XeTESS[self.TeTESS[subelem,:],:],
                                    Te = self.Te,
                                    PlasmaLSe = None, 
                                    VacVessLSe= None)
            # ASSIGN A REGION TO EACH SUBELEMENT
            self.SubElements[subelem].Dom = self.DomTESS[subelem]
            
        ######### MODIFIED QUADRATURE TO INTEGRATE OVER SUBELEMENTS
        # 1. MAP THE PHYSICAL INTERFACE ON THE REFERENCE ELEMENT
        self.XIeint = np.zeros(np.shape(self.InterEdges[0].Xeint))
        for i in range(2):
            self.XIeint[i,:] = self.InverseMapping(self.InterEdges[0].Xeint[i,:])
            
        # 2. DO TESSELLATION ON REFERENCE ELEMENT
        self.XIeTESS, self.TeTESSREF = self.ReferenceElementTessellation(self.XIeint)
        
        # 3. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS AND EVALUATE INTEGRATION ENTITIES ON THEM
        for i, subelem in enumerate(self.SubElements):
            #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES (2D)
            XIg2DFEM, subelem.Wg2D, subelem.Ng2D = GaussQuadrature(subelem.ElType,Order)
            # EVALUATE SUBELEMENTAL REFERENCE SHAPE FUNCTIONS 
            N2D, foo, foo = EvaluateReferenceShapeFunctions(XIg2DFEM, subelem.ElType, subelem.ElOrder, subelem.n)
        
            # MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS  ->> MODIFIED 2D QUADRATURE FOR SUBELEMENTS
            subelem.XIg2D = N2D @ self.XIeTESS[self.TeTESSREF[i,:]]
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS ON MODIFIED REFERENCE QUADRATURE
            subelem.N, subelem.dNdxi, subelem.dNdeta = EvaluateReferenceShapeFunctions(subelem.XIg2D, self.ElType, self.ElOrder, self.n)
            # MAPP MODIFIED REFERENCE QUADRATURE ON PHYSICAL ELEMENT
            subelem.Xg2D = subelem.N @ self.Xe
            
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON MODIFIED QUADRATURES NODES
            subelem.invJg = np.zeros([subelem.Ng2D,subelem.dim,subelem.dim])
            subelem.detJg = np.zeros([subelem.Ng2D])
            Rmeansub = np.sum(subelem.Xe[:,0])/subelem.n   # mean subelemental radial position
            for ig in range(subelem.Ng2D):
                #subelem.invJg[ig,:,:], subelem.detJg[ig] = Jacobian(subelem.Xe[:,0],subelem.Xe[:,1],subelem.dNdxi[ig,:],subelem.dNdeta[ig,:])
                subelem.invJg[ig,:,:], subelem.detJg[ig] = Jacobian(self.Xe[:,0],self.Xe[:,1],subelem.dNdxi[ig,:],subelem.dNdeta[ig,:])   #### MIRAR ESTO EN FORTRAN !!
                subelem.detJg[ig] *= 2*np.pi*Rmeansub   # ACCOUNT FOR AXISYMMETRICAL
                
        ######### MODIFIED QUADRATURE TO INTEGRATE OVER ELEMENTAL INTERFACES
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1DFEM, Wg1D, Ng1D = GaussQuadrature(0,Order)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1DFEM, 0, Order-1, 2)
                
        for edge in range(self.Neint):
            self.InterEdges[edge].Ngaussint = Ng1D
            self.InterEdges[edge].Wgint = Wg1D
            self.InterEdges[edge].detJgint = np.zeros([self.InterEdges[edge].Ngaussint])
            # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE ->> MODIFIED 1D QUADRATURE FOR INTERFACE
            self.InterEdges[edge].XIgint = N1D @ self.XIeint
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON INTERFACE MODIFIED QUADRATURE
            self.InterEdges[edge].Nint, self.InterEdges[edge].dNdxiint, self.InterEdges[edge].dNdetaint = EvaluateReferenceShapeFunctions(self.InterEdges[edge].XIgint, self.ElType, self.ElOrder, self.n)
            # MAPP REFERENCE INTERFACE MODIFIED QUADRATURE ON PHYSICAL ELEMENT 
            self.InterEdges[edge].Xgint = N1D @ self.InterEdges[edge].Xeint
            
            Rmeanint = np.mean(self.InterEdges[edge].Xeint[:,0])   # mean interface radial position
            for ig in range(self.InterEdges[edge].Ngaussint):
                self.InterEdges[edge].detJgint[ig] = Jacobian1D(self.InterEdges[edge].Xeint[:,0],self.InterEdges[edge].Xeint[:,1],dNdxi1D[ig,:])  
                self.InterEdges[edge].detJgint[ig] *= 2*np.pi*Rmeanint   # ACCOUNT FOR AXISYMMETRICAL    
        return 
    
    
    ##################################################################################################
    ################################ ELEMENTAL INTEGRATION ###########################################
    ##################################################################################################
    
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
            for i in range(len(self.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(self.Te)):   # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) *(Jacobiano*2pi*rad) ]  
                    LHS[self.Te[i],self.Te[j]] -= np.transpose((self.invJg[ig,:,:]@Ngrad[:,i]))@(self.invJg[ig,:,:]@Ngrad[:,j])*self.detJg[ig]*self.Wg2D[ig]
                    ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j *(Jacobiano*2pi*rad) ]  ONLY RESPECT TO R
                    LHS[self.Te[i],self.Te[j]] -= (1/self.Xg2D[ig,0])*self.N[ig,j] * (self.invJg[ig,0,:]@Ngrad[:,i])*self.detJg[ig]*self.Wg2D[ig]
                # COMPUTE RHS VECTOR TERMS [ (source term)*N_i*(Jacobiano *2pi*rad) ]
                RHS[self.Te[i]] += SourceTermg[ig] * self.N[ig,i] *self.detJg[ig]*self.Wg2D[ig]
        return 
    
    
    def IntegrateElementalInterfaceTerms(self,beta,LHS,RHS):
        """ Input: - PHI_g: Interface condition, evaluated at physical gauss integration nodes
                   - beta: Nitsche's method penalty parameter
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector 
                    """
    
        # LOOP OVER EDGES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        for edge in range(self.Neint):
            # ISOLATE INTERFACE EDGE
            EDGE = self.InterEdges[edge]
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(EDGE.Ngaussint):  
                # SHAPE FUNCTIONS GRADIENT
                Ngrad = np.array([EDGE.dNdxiint[ig,:],EDGE.dNdetaint[ig,:]])
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                for i in range(len(self.Te)):  # ROWS ELEMENTAL MATRIX
                    for j in range(len(self.Te)):  # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) *(Jacobiano*2pi*rad) ]  
                        LHS[self.Te[i],self.Te[j]] += EDGE.Nint[ig,i] * EDGE.NormalVec @ Ngrad[:,j] * EDGE.detJgint[ig] * EDGE.Wgint[ig]
                        ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) *(Jacobiano*2pi*rad) ]
                        LHS[self.Te[i],self.Te[j]] += EDGE.NormalVec @ Ngrad[:,i]*(EDGE.Nint[ig,j] * EDGE.detJgint[ig] * EDGE.Wgint[ig])
                        ### PENALTY TERM   [ beta * (N_i*N_j) *(Jacobiano*2pi*rad) ]
                        LHS[self.Te[i],self.Te[j]] += beta * EDGE.Nint[ig,i] * EDGE.Nint[ig,j] * EDGE.detJgint[ig] * EDGE.Wgint[ig]
                    # COMPUTE RHS VECTOR TERMS 
                    ### SYMMETRIC NITSCHE'S METHOD TERM  [ PHI_D * (n dot nabla(N_i)) * (Jacobiano *2pi*rad) ]
                    RHS[self.Te[i]] +=  EDGE.PHI_g[ig] * EDGE.NormalVec @ Ngrad[:,i] * EDGE.detJgint[ig] * EDGE.Wgint[ig]
                    ### PENALTY TERM   [ beta * N_i * PHI_D *(Jacobiano*2pi*rad) ]
                    RHS[self.Te[i]] +=  beta * EDGE.PHI_g[ig] * EDGE.Nint[ig,i] * EDGE.detJgint[ig] * EDGE.Wgint[ig]
        return 
    
    
    ##################################################################################################
    ################################ ELEMENT CHARACTERISATION ########################################
    ##################################################################################################
        
def ElementalNumberOfEdges(elemType):
    match elemType:
        case 0:
            numedges = 1
        case 1:
            numedges = 3
        case 2:  
            numedges = 4
    return numedges     

    
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
    