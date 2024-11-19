# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Author: Pau Manyer Fuertes
# Email: pau.manyer@bsc.es
# Date: October 2024
# Institution: Barcelona Supercomputing Center (BSC)
# Department: Computer Applications in Science and Engineering (CASE)
# Research Group: Nuclear Fusion  


# This script contains the definition for class ELEMENT, an object which 
# embodies the cell elements constituing the mesh. For each ELEMENT object,
# coordinates, shape functions, numerical integration quadratures... data is 
# stored and defines the object. Several elemental methods are also defined 
# inside this class.


from src.GaussQuadrature import *
from src.ShapeFunctions import *
from scipy import optimize
from itertools import chain
from src.Segment import *

class Element:
    
    ##################################################################################################
    ################################ ELEMENT INITIALISATION ##########################################
    ##################################################################################################
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,PlasmaLSe,VacVessLSe):
        
        self.index = index                                              # GLOBAL INDEX ON COMPUTATIONAL MESH
        self.ElType = ElType                                            # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
        self.ElOrder = ElOrder                                          # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.numedges = ElementalNumberOfEdges(ElType)                  # ELEMENTAL NUMBER OF EDGES
        self.n, self.nedge = ElementalNumberOfNodes(ElType, ElOrder)    # NUMBER OF NODES PER ELEMENT, PER ELEMENTAL EDGE
        self.Xe = Xe                                                    # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.dim = len(Xe[0,:])                                         # SPATIAL DIMENSION
        self.Te = Te                                                    # ELEMENTAL CONNECTIVITIES
        self.PlasmaLSe = PlasmaLSe                                      # ELEMENTAL NODAL PLASMA REGION LEVEL-SET VALUES
        self.VacVessLSe = VacVessLSe                                    # ELEMENTAL NODAL VACUUM VESSEL FIRST WALL LEVEL-SET VALUES
        self.PSIe = np.zeros([self.n])                                  # ELEMENTAL NODAL PSI VALUES
        self.Dom = None                                                 # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; 0: "PLASMA INTERFACE"; +1: "VACUUM" ; +2: FIRST WALL ; +3: "EXTERIOR")
        
        # INTEGRATION QUADRATURES ENTITIES
        self.ng = None              # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 2D GAUSS QUADRATURE
        self.XIg = None             # GAUSS INTEGRATION NODES 
        self.Wg = None              # GAUSS INTEGRATION WEIGTHS
        self.Ng = None              # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None          # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.dNdetag = None         # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT GAUSS INTEGRATION NODES
        self.Xg = None              # PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM 2D REFERENCE ELEMENT
        self.invJg = None           # INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES 
        
        ### ATTRIBUTES FOR INTERFACE ELEMENTS
        self.Neint = None           # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.InterfApprox = None    # ARRAY CONTAINING THE ELEMENTAL EDGES/CUTS CORRESPONDING TO INTERFACES
        self.Nesub = None           # NUMBER OF SUBELEMENTS GENERATED IN TESSELLATION
        return
    
    
    ##################################################################################################
    #################################### ELEMENTAL MAPPING ###########################################
    ##################################################################################################
    
    
    def Mapping(self,Xi):
        """ This function implements the mapping corresponding to the transformation from natural to physical coordinates. 
        That is, given a point in the reference element with coordinates Xi, this function returns the coordinates X of the corresponding point mapped
        in the physical element with nodal coordinates Xe. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric equations. 
        
        Input: - Xg: coordinates of point in reference space for which to compute the coordinate in physical space
               - Xe: nodal coordinates of physical element
        Output: - X: coodinates of mapped point in reference element """
        
        N, foo, foo = EvaluateReferenceShapeFunctions(Xi, self.ElType, self.ElOrder)
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
    
    def ElementalInterpolationPHYSICAL(self,X,Fe):
        """ Interpolate field F with nodal values Fe on point X using elemental shape functions. """
        XI = self.InverseMapping(X)
        return self.ElementalInterpolationREFERENCE(self,XI,Fe)
    
    def ElementalInterpolationREFERENCE(self,XI,Fe):
        """ Interpolate field F with nodal values Fe on point X using elemental shape functions. """
        F = 0
        for i in range(self.n):
            N, foo, foo = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            F += N*Fe[i]
        return F
    
    
    ##################################################################################################
    ######################### CUT ELEMENTS INTERFACE APPROXIMATION ###################################
    ##################################################################################################
    
    def InterfaceApproximation(self,interface_index):
        # READ LEVEL-SET NODAL VALUES
        if self.Dom == 0:  # PLASMA/VACUUM INTERFACE ELEMENT
            LSe = self.PlasmaLSe  
        if self.Dom == 2:  # VACUUM VESSEL FIRST WALL ELEMENT
            LSe = self.VacVessLSe
        
        # OBTAIN REFERENCE ELEMENT COORDINATES
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)

        # OBTAIN INTERSECTION COORDINATES FOR EACH EDGE:
        self.Neint = 1
        self.InterfApprox = [InterfaceApprox(index = interface_index,
                                            Nsegments = self.ElOrder) for interf in range(self.Neint)]
        
        for INTERFACE in self.InterfApprox:
            # FIND POINTS ON INTERFACE USING ELEMENTAL INTERPOLATION
            #### INTERSECTION WITH EDGES
            XIintEND = np.zeros([2,2])
            INTERFACE.ElIntNodes = np.zeros([2,2],dtype=int)
            k = 0
            for i in range(self.nedge):  # Loop over elemental edges
                # Check for sign change along the edge
                inode = i
                jnode = (i + 1) % self.nedge
                if LSe[inode] * LSe[jnode] < 0:
                    INTERFACE.ElIntNodes[k,:] = [inode,jnode]
                    if abs(XIe[jnode,0]-XIe[inode,0]) < 1e-6: # VERTICAL EDGE
                        #### DEFINE CONSTRAINT PHI FUNCTION
                        xi = XIe[inode,0]
                        def PHIedge(eta):
                            X = np.array([xi,eta]).reshape((1,2))
                            N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                            return N@LSe
                        #### FIND INTERSECTION POINT:
                        Eta0 = 1/2  # INITIAL GUESS FOR ROOT SOLVER
                        sol = optimize.root(PHIedge, Eta0)
                        XIintEND[k,:] = [xi, sol.x]
                        k += 1
                    else:
                        def edgeconstraint(xi):
                            # FUNCTION DEFINING THE CONSTRAINT ON THE ELEMENTAL EDGE
                            m = (XIe[jnode,1]-XIe[inode,1])/(XIe[jnode,0]-XIe[inode,0])
                            eta = m*(xi-XIe[inode,0])+XIe[inode,1]
                            return eta
                        def PHIedge(xi):
                            X = np.array([xi,edgeconstraint(xi)]).reshape((1,2))
                            N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                            return N@LSe
                        #### FIND INTERSECTION POINT:
                        Xi0 = 1/2  # INITIAL GUESS FOR ROOT SOLVER
                        sol = optimize.root(PHIedge, Xi0)
                        XIintEND[k,:] = [sol.x, edgeconstraint(sol.x)]
                        k += 1
                    
            #### HIGH-ORDER INTERFACE NODES
            # IN THIS CASE, WITH USE THE REGULARITY OF THE REFERENCE TRIANGLE TO FIND THE NODES
            # LYING ON THE INTERFACE INSIDE THE ELEMENT. SIMILARLY TO THE INTERSECTION NODES ON THE
            # ELEMENTAL EDGES, EACH INTERIOR NODE CAN BE FOUND BY IMPOSING TWO CONDITIONS:
            #    - PHI = 0
            #    - NODE ON LINE DIVIDING THE 
            def PHI(X):
                N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                return N@LSe

            def fun(X):
                F = np.zeros([X.shape[0]])
                # SEPARATE GUESS VECTOR INTO INDIVIDUAL NODAL COORDINATES
                XHO = X.reshape((self.ElOrder-1,2)) 
                # PHI = 0 ON NODES
                for inode in range(self.ElOrder-1):
                    F[inode] = PHI(XHO[inode,:].reshape((1,2)))
                # EQUAL DISTANCES BETWEEN INTERFACE NODES
                if self.ElOrder == 2:
                    F[-1] = np.linalg.norm(XIintEND[0,:]-X)-np.linalg.norm(XIintEND[1,:]-X)
                if self.ElOrder == 3:
                    #### FIRST INTERVAL
                    F[self.ElOrder-1] = np.linalg.norm(XIintEND[0,:]-XHO[0,:])-np.linalg.norm(XHO[0,:]-XHO[1,:])
                    #### LAST INTERVAL
                    F[-1] = np.linalg.norm(XIintEND[1,:]-XHO[-1,:])-np.linalg.norm(XHO[-1,:]-XHO[-2,:])
                #### INTERIOR INTERVALS
                if self.ElOrder > 3:
                    for intv in range(self.ElOrder-3):
                        F[self.ElOrder+intv] = np.linalg.norm(XHO[intv+1,:]-XHO[intv+2,:]) - np.linalg.norm(XHO[intv+2,:]-XHO[intv+3,:])
                return F

            # PREPARE INITIAL GUESS
            X0 = np.zeros([(self.ElOrder-1)*2])
            for inode in range(1,self.ElOrder):
                X0[2*(inode-1):2*inode] = XIintEND[0,:] + np.array([(XIintEND[1,0]-XIintEND[0,0]),(XIintEND[1,1]-XIintEND[0,1])])*inode/self.ElOrder
            X0 = X0.reshape((1,(self.ElOrder-1)*2))
            # COMPUTE HIGH-ORDER INTERFACE NODES COORDINATES
            sol = optimize.root(fun, X0)
            # STORE SOLUTION NODES
            XIintINT = np.zeros([self.ElOrder-1,2])
            for inode in range(self.ElOrder-1):
                XIintINT[inode,:] = np.reshape(sol.x, (self.ElOrder-1,2))[inode,:]

            ##### STORE INTERFACE APPROXIMATION DATA IN INTERFACE OBJECT 
            ## CONCATENATE INTERFACE NODES
            INTERFACE.XIint = np.concatenate((XIintEND,XIintINT),axis=0)
            
            ## MAP BACK TO PHYSICAL SPACE
            # EVALUATE REFERENCE SHAPE FUNCTIONS AT POINTS TO MAP (INTERFACE NODES)
            Nint, foo, foo = EvaluateReferenceShapeFunctions(INTERFACE.XIint, self.ElType, self.ElOrder)
            # COMPUTE SCALAR PRODUCT
            INTERFACE.Xint = Nint@self.Xe
            
            ## ASSOCIATE ELEMENTAL CONNECTIVITY TO INTERFACE SEGMENTS
            lnods = [0,np.arange(2,self.ElOrder+1),1]
            INTERFACE.Tint = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in lnods))
        
            #### GENERATE SEGMENT OBJECTS AND STORE INTERFACE DATA
            INTERFACE.Segments = [Segment(index = iseg,
                                        ElOrder = self.ElOrder,   
                                        Xseg = INTERFACE.Xint[INTERFACE.Tint[iseg:iseg+2]]) 
                                        for iseg in range(INTERFACE.Nsegments)]   
            for iseg, SEGMENT in enumerate(INTERFACE.Segments):
                # STORE REFERENCE APRIORI NODES 
                SEGMENT.XIseg = INTERFACE.XIint[INTERFACE.Tint[iseg:iseg+2]]
                # COMPUTE INNER HIGH-ORDER NODES IN EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION (BOTH REFERENCE AND PHYSICAL SPACE)
                XsegHO = np.zeros([SEGMENT.n,SEGMENT.dim])
                XIsegHO = np.zeros([SEGMENT.n,SEGMENT.dim])
                XsegHO[:2,:] = SEGMENT.Xseg
                XIsegHO[:2,:] = SEGMENT.XIseg
                dx = (SEGMENT.Xseg[1,0]-SEGMENT.Xseg[0,0])/(SEGMENT.n-1)
                dy = (SEGMENT.Xseg[1,1]-SEGMENT.Xseg[0,1])/(SEGMENT.n-1)
                dxi = (SEGMENT.XIseg[1,0]-SEGMENT.XIseg[0,0])/(SEGMENT.n-1)
                deta = (SEGMENT.XIseg[1,1]-SEGMENT.XIseg[0,1])/(SEGMENT.n-1)
                for iinnernode in range(2,SEGMENT.n):
                    XsegHO[iinnernode,:] = [SEGMENT.Xseg[0,0]+(iinnernode-1)*dx,
                                            SEGMENT.Xseg[0,1]+(iinnernode-1)*dy]
                    XIsegHO[iinnernode,:] = [SEGMENT.XIseg[0,0]+(iinnernode-1)*dxi,
                                            SEGMENT.XIseg[0,1]+(iinnernode-1)*deta]
                # STORE HIGH-ORDER SEGMENT ELEMENTS CONFORMING HIGH-ORDER INTERFACE APPROXIMATION
                SEGMENT.Xseg = XsegHO
                SEGMENT.XIseg = XIsegHO
                
        return 
    
    
    def ComputationalDomainBoundaryEdges(self,Tbound):
        """ This function finds for each element the edges lying on the computational domain's boundary. The different elemental attributes are set-up accordingly.
        
        Input: - Tbound: # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        
        COMPUTED ATTRIBUTES:
                * FOR ELEMENT
                    - Neint: NUMBER OF INTERFACES IN ELEMENT
                * FOR INTERFACE SEGMENT
                    - Xint: PHYSICAL INTERFACE SEGMENT VERTICES COORDINATES 
                    - inter_edges: LOCAL INDICES OF VERTICES WHERE SEGMENT ENDS ARE LOCATED
                """
        
        # OBTAIN REFERENCE ELEMENT COORDINATES
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        
        #### LOOK WHICH BOUNDARIES ARE ASSOCIATED TO THE ELEMENT
        interface = np.where(Tbound[:,-1] == self.index)[0]         # GLOBAL INDEX FOR COMPUTATIONAL DOMAIN'S BOUNDARY ELEMENTAL EDGE
        self.Neint = len(interface)                                 # NUMBER OF ELEMENTAL EDGES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        
        #### GENERATE INTERFACE OBJECT
        self.InterfApprox = [InterfaceApprox(index = interface[interf],
                                            Nsegments = 1) for interf in range(self.Neint)]
           
        for interf, INTERFACE in enumerate(self.InterfApprox):
            # GENERATE SEGMENT OBJECTS AND STORE INTERFACE DATA
            INTERFACE.Segments = [Segment(index = iseg,
                                        ElOrder = self.ElOrder,   
                                        Xseg = np.zeros([2,self.dim])) for iseg in range(INTERFACE.Nsegments)] 
            # FIND LOCAL INDEXES OF NODES ON EDGE 
            INTERFACE.ElIntNodes = np.zeros([len(Tbound[0,:-1])], dtype=int)
            for i in range(len(Tbound[0,:-1])):
                INTERFACE.ElIntNodes[i] = np.where(Tbound[interface[interf],i] == self.Te)[0][0]
            # COORDINATES OF NODES ON EDGE (PHYSICAL SPACE)
            INTERFACE.Xint = self.Xe[INTERFACE.ElIntNodes,:]
            # COORDINATES OF NODES ON EDGE (REFERENCE SPACE)
            INTERFACE.XIint = XIe[INTERFACE.ElIntNodes,:]
            INTERFACE.Tint = np.array([[0,1]], dtype=int)
            # STORE DATA ON SEGMENT OBJECT
            for SEGMENT in INTERFACE.Segments:
                SEGMENT.Xseg = INTERFACE.Xint
                SEGMENT.XIseg = INTERFACE.XIint
                
        return 
    
    
    ##################################################################################################
    ##################################### INTERFACE NORMALS ##########################################
    ##################################################################################################
    
    def InterfaceNormal(self):
        """ This function computes the interface normal vector pointing outwards. """
        
        # LEVEL-SET NODAL VALUES
        if self.Dom == 0:  # ELEMENT CONTAINING PLASMA/VACUUM INTERFACE
            LSe = self.PlasmaLSe
        elif self.Dom == 2:  # ELEMENT CONTAINING VACUUM VESSEL FIRST WALL
            LSe = self.VacVessLSe
        # COMPUTE THE NORMAL VECTOR FOR EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION
        for INTERFACE in self.InterfApprox:
            for SEGMENT in INTERFACE.Segments:
                #### PREPARE TEST NORMAL VECTOR IN PHYSICAL SPACE
                dx = SEGMENT.Xseg[1,0] - SEGMENT.Xseg[0,0]
                dy = SEGMENT.Xseg[1,1] - SEGMENT.Xseg[0,1]
                ntest_xy = np.array([-dy, dx]) 
                ntest_xy = ntest_xy/np.linalg.norm(ntest_xy) 
                
                #### PERFORM THE TEST IN REFERENCE SPACE
                # PREPARE TEST NORMAL VECTOR IN REFERENCE SPACE
                dxi = SEGMENT.XIseg[1,0] - SEGMENT.XIseg[0,0]
                deta = SEGMENT.XIseg[1,1] - SEGMENT.XIseg[0,1]
                ntest_xieta = np.array([-deta, dxi])                     # test this normal vector
                ntest_xieta = ntest_xieta/np.linalg.norm(ntest_xieta)    # normalize
                XIsegmean = np.mean(SEGMENT.XIseg, axis=0)               # mean point on interface
                XItest = XIsegmean + ntest_xieta                         # physical point on which to test the Level-Set 
                # INTERPOLATE LEVEL-SET IN REFERENCE SPACE
                LStest = self.ElementalInterpolationREFERENCE(XItest,LSe)
                # CHECK SIGN OF LEVEL-SET 
                if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
                    SEGMENT.NormalVec = ntest_xy
                else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
                    SEGMENT.NormalVec = -1*ntest_xy
        return 
    
    def ComputationalDomainBoundaryNormal(self,Xmax,Xmin,Ymax,Ymin):
        """ This function computes the boundary edge(s) normal vector(s) pointing outwards. """
        
        # COMPUTE THE NORMAL VECTOR FOR EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION
        for INTERFACE in self.InterfApprox:
            for SEGMENT in INTERFACE.Segments:
                dx = SEGMENT.Xseg[1,0] - SEGMENT.Xseg[0,0]
                dy = SEGMENT.Xseg[1,1] - SEGMENT.Xseg[0,1]
                ntest = np.array([-dy, dx])                 # test this normal vector
                ntest = ntest/np.linalg.norm(ntest)         # normalize
                Xsegmean = np.mean(SEGMENT.Xseg, axis=0)    # mean point on interface
                Xtest = Xsegmean + 3*ntest                  # physical point on which to test the Level-Set 
                
                # CHECK IF TEST POINT IS OUTSIDE COMPUTATIONAL DOMAIN
                if Xtest[0] < Xmin or Xmax < Xtest[0] or Xtest[1] < Ymin or Ymax < Xtest[1]:  
                    SEGMENT.NormalVec = ntest
                else: 
                    SEGMENT.NormalVec = -1*ntest
        return
    
    
    ##################################################################################################
    ################################ ELEMENTAL TESSELLATION ##########################################
    ##################################################################################################
        
    @staticmethod
    def HO_TRI_interf(XeLIN,ElOrder,XintHO,interfedge):
        nedge = len(XeLIN[:,0])
        XeHO = XeLIN.copy()
        # MAKE IT HIGH-ORDER:
        for iedge in range(nedge):
            inode = iedge
            jnode = (iedge+1)%nedge
            # EDGE HIGH-ORDER NODES
            if interfedge == iedge:
                XeHO = np.concatenate((XeHO,XintHO[2:,:]), axis=0)
            else:
                for k in range(1,ElOrder):
                    HOnode = np.array([XeLIN[inode,0]+((XeLIN[jnode,0]-XeLIN[inode,0])/ElOrder)*k,XeLIN[inode,1]+((XeLIN[jnode,1]-XeLIN[inode,1])/ElOrder)*k])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        # INTERIOR HIGH-ORDER NODES:
        if ElOrder == 3:
            HOnode = np.array([np.mean(XeHO[:,0]),np.mean(XeHO[:,1])])
            XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        return XeHO

    @staticmethod
    def HO_QUA_interf(XeLIN,ElOrder,XintHO,interfedge):
        nedge = len(XeLIN[:,0])
        XeHO = XeLIN.copy()
        for iedge in range(nedge):
            inode = iedge
            jnode = (iedge+1)%nedge
            # EDGE HIGH-ORDER NODES
            if interfedge == iedge:
                XeHO = np.concatenate((XeHO,XintHO[2:,:]), axis=0)
            else:
                for k in range(1,ElOrder):
                    HOnode = np.array([XeLIN[inode,0]+((XeLIN[jnode,0]-XeLIN[inode,0])/ElOrder)*k,XeLIN[inode,1]+((XeLIN[jnode,1]-XeLIN[inode,1])/ElOrder)*k])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        # INTERIOR HIGH-ORDER NODES:
        if ElOrder == 2:
            HOnode = np.array([np.mean(XeHO[:,0]),np.mean(XeHO[:,1])])
            XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        elif ElOrder == 3:
            for k in range(1,ElOrder):
                dx = (XeHO[12-k,0]-XeHO[5+k,0])/ElOrder
                dy = (XeHO[12-k,1]-XeHO[5+k,1])/ElOrder
                for j in range(1,ElOrder):
                    if k == 1:
                        HOnode = XeHO[11,:] - np.array([dx*j,dy*j])
                    elif k == 2:
                        HOnode = XeHO[7,:] + np.array([dx*j,dy*j])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        return XeHO


    def ReferenceElementTessellation(self):
        """ This function performs the TESSELLATION of a HIGH-ORDER REFERENCE ELEMENT with interface nodal coordinates XIeintHO
        
            Output: XIeTESSHO: High-order subelemental nodal coordinates matrix for each child element generated in the tessellation,
                               such that:
                                            XIeTESSHO = [[[ xi00, eta00 ],
                                                          [ xi01, eta01 ],      NODAL COORDINATE MATRIX
                                                                ....    ],         FOR SUBELEMENT 0
                                                          [ xi0n, eta0n ]],
                                                          
                                                         [[ xi10, eta10 ],
                                                          [ xi11, eta11 ],      NODAL COORDINATE MATRIX
                                                                ....    ],         FOR SUBELEMENT 1
                                                          [ xi1n, eta1n ]],
                                                          
                                                                ....    ]
        """
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        # HOWEVER, WHEN LOOKING FOR THE LINEAR APPROXIMATION OF THE PHYSICAL INTERFACE THIS PROCESS IS ALREADY DONE, THEREFORE WE CAN SKIP IT. 
        # IF INPUT Xemod IS PROVIDED, THE TESSELLATION IS DONE ACCORDINGLY TO ADAPTED NODAL MATRIX Xemod WHICH IS ASSUMED TO HAS THE PREVIOUSLY DESCRIBED STRUCTURE.
        # IF NOT, THE COMMON NODE IS DETERMINED (THIS IS THE CASE FOR INSTANCE WHEN THE REFERENCE ELEMENT IS TESSELLATED).

        XIeLIN = ReferenceElementCoordinates(self.ElType,1)
        INTERFACE = self.InterfApprox[0]
        edgenodes = INTERFACE.ElIntNodes

        if self.ElType == 1:  # TRIANGULAR ELEMENT
            Nesub = 3
            SubElType = 1
            distance = np.zeros([2])
            edgenode = np.zeros([2],dtype=int)
            commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop() # COMMON NODE TO INTERSECTED EDGES
            # LOOK FOR NODE ON EDGE WHERE INTERSECTION POINT LIES BUT OTHER THAN COMMON NODE AND COMPUTE DISTANCE
            for i in range(2):
                edgenodeset = set(edgenodes[i,:])
                edgenodeset.remove(commonnode)
                edgenode[i] = edgenodeset.pop()
                distance[i] = np.linalg.norm(INTERFACE.XIint[i,:]-XIeLIN[edgenode[i],:])
            
            XIeTESSLIN = list()
            interfedge = [1,1,-1]
            XIeTESSLIN.append(np.concatenate((XIeLIN[int(commonnode),:].reshape((1,2)),INTERFACE.XIint[:2,:]),axis=0))
            if distance[0] < distance[1]:
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenode[1],:].reshape((1,2)),INTERFACE.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[[edgenode[0],edgenode[1]],:],INTERFACE.XIint[0,:].reshape((1,2))),axis=0))
            else:
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenode[0],:].reshape((1,2)),INTERFACE.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[[edgenode[1],edgenode[0]],:],INTERFACE.XIint[1,:].reshape((1,2))),axis=0))
            
            # TURN LINEAR SUBELEMENTS INTO HIGH-ORDER SUBELEMENTS
            XIeTESSHO = list()
            for isub in range(Nesub):
                XIeHO = self.HO_TRI_interf(XIeTESSLIN[isub],self.ElOrder,INTERFACE.XIint,interfedge[isub])
                XIeTESSHO.append(XIeHO)
                
            
        elif self.ElType == 2:  # QUADRILATERAL ELEMENT
            # LOOK FOR TESSELLATION CONFIGURATION BY USING SIGN OF prod(LSe)
                    #  -> IF prod(LSe) > 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO 2 CHILD QUADRILATERAL ELEMENTS
                    #  -> IF prod(LSe) < 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO PENTAGON AND TRIANGLE -> PENTAGON IS SUBDIVIDED INTO 3 SUBTRIANGLES
            
            # LEVEL-SET NODAL VALUES
            if self.Dom == 0:  # ELEMENT CONTAINING PLASMA/VACUUM INTERFACE
                LSe = self.PlasmaLSe
            elif self.Dom == 2:  # ELEMENT CONTAINING VACUUM VESSEL FIRST WALL
                LSe = self.VacVessLSe
            
            if np.prod(LSe[:self.nedge]) > 0:  # 2 SUBQUADRILATERALS
                Nesub = 2
                SubElType = 2
            
                interfedge = [1,1]
                XIeTESSLIN = list()
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenodes[0,0],:].reshape((1,2)),INTERFACE.XIint[:2,:],XIeLIN[edgenodes[1,1],:].reshape((1,2))),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenodes[0,1],:].reshape((1,2)),INTERFACE.XIint[:2,:],XIeLIN[edgenodes[1,0],:].reshape((1,2))),axis=0))
                
                # TURN LINEAR TRIANGULAR SUBELEMENTS INTO HIGH-ORDER TRIANGULAR SUBELEMENTS
                XIeTESSHO = list()
                for isub in range(Nesub):
                    XIeHO = self.HO_QUA_interf(XIeTESSLIN[isub],self.ElOrder,INTERFACE.XIint,interfedge[isub])
                    XIeTESSHO.append(XIeHO)
                
            else:  # 4 SUBTRIANGLES
                Nesub = 4
                SubElType = 1
                # LOOK FOR COMMON NODE
                edgenode = np.zeros([2],dtype=int)
                distance = np.zeros([2])
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
                        
                XIeTESSLIN = list()
                interfedge = [1,1,-1,-1]
                XIeTESSLIN.append(np.concatenate((XIeLIN[int(commonnode),:].reshape((1,2)),INTERFACE.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[oppositenode,:].reshape((1,2)),INTERFACE.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((INTERFACE.XIint[0,:].reshape((1,2)),XIeLIN[[edgenode[0],oppositenode],:]),axis=0))
                XIeTESSLIN.append(np.concatenate((INTERFACE.XIint[1,:].reshape((1,2)),XIeLIN[[edgenode[1],oppositenode],:]),axis=0))
                
                # TURN LINEAR TRIANGULAR SUBELEMENTS INTO HIGH-ORDER TRIANGULAR SUBELEMENTS
                XIeTESSHO = list()
                for isub in range(Nesub):
                    XIeHO = self.HO_TRI_interf(XIeTESSLIN[isub],self.ElOrder,INTERFACE.XIint,interfedge[isub])
                    XIeTESSHO.append(XIeHO)
                
        return Nesub, SubElType, XIeTESSHO, interfedge
    
        
    ##################################################################################################
    ############################### ELEMENTAL NUMERICAL QUADRATURES ##################################
    ##################################################################################################
        
    def ComputeStandardQuadrature2D(self,NumQuadOrder):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to integrations in 2D for elements which ARE NOT CUT BY ANY INTERFACE. Hence, 
        in such elements the standard FEM integration methodology is applied (STANDARD REFERENCE SHAPE FUNCTIONS EVALUATED AT STANDARD GAUSS INTEGRATION NODES). 
        
            Input: NumQuadOrder: Numerical integration Quadrature Order 
            """
        
        # COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE IN 2D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES 
        self.XIg, self.Wg, self.ng = GaussQuadrature(self.ElType,NumQuadOrder)
        
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        self.Ng, self.dNdxig, self.dNdetag = EvaluateReferenceShapeFunctions(self.XIg, self.ElType, self.ElOrder)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES INVERSE MATRIX 
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE ELEMENT
          
        # COMPUTE MAPPED GAUSS NODES
        self.Xg = self.Ng @ self.Xe       
        # COMPUTE JACOBIAN INVERSE AND DETERMINANT
        self.invJg = np.zeros([self.ng,self.dim,self.dim])
        self.detJg = np.zeros([self.ng])
        for ig in range(self.ng):
            self.invJg[ig,:,:], self.detJg[ig] = Jacobian(self.Xe,self.dNdxig[ig,:],self.dNdetag[ig,:])
            
        return    
            
    def ComputeComputationalDomainBoundaryQuadrature(self, NumQuadOrder):       
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to integrations in 1D for elements which ARE NOT CUT by the interface. Hence, 
        in such elements the standard FEM integration methodology is applied (STANDARD REFERENCE SHAPE FUNCTIONS EVALUATED AT STANDARD GAUSS INTEGRATION NODES). 
        
            Input: NumQuadOrder: Numerical integration Quadrature Order  
        """   
         
        # COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE IN 1D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1D, Wg1D, Ng1D = GaussQuadrature(0,NumQuadOrder)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1D, 0, self.nedge-1)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 1D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE 1D ELEMENT
        
        for INTERFACE in self.InterfApprox:
            for SEGMENT in INTERFACE.Segments:
                SEGMENT.ng = Ng1D
                SEGMENT.Wg = Wg1D
                SEGMENT.detJg = np.zeros([SEGMENT.ng])
                # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON REFERENCE VACUUM VESSEL FIRST WALL EDGE 
                SEGMENT.XIg = N1D @ SEGMENT.XIseg
                # EVALUATE 2D REFERENCE SHAPE FUNCTION ON REFERENCE VACUUM VESSEL FIRST WALL EDGE GAUSS NODES
                SEGMENT.Ng, SEGMENT.dNdxig, SEGMENT.dNdetag = EvaluateReferenceShapeFunctions(SEGMENT.XIg, self.ElType, self.ElOrder)
                # COMPUTE MAPPED GAUSS NODES
                SEGMENT.Xg = N1D @ SEGMENT.Xseg
                # COMPUTE JACOBIAN INVERSE AND DETERMINANT
                for ig in range(SEGMENT.ng):
                    SEGMENT.detJg[ig] = Jacobian1D(SEGMENT.Xseg,dNdxi1D[ig])  
                
        return
    
    
    def ComputeAdaptedQuadratures(self,NumQuadOrder):
        """ This function computes the NUMERICAL INTEGRATION QUADRATURES corresponding to a 2D and 1D integration for elements which ARE CUT BY AN INTERFACE. 
        In this case, an adapted quadrature is computed by modifying the standard approach.  
        
        Input: NumQuadOrder: Numerical integration Quadrature Order
            
            # IN ORDER TO COMPUTE THE ADAPTED QUADRATURES, WE NEED TO:
            #    - ADAPTED QUADRATURE TO INTEGRATE OVER SUBELEMENTS:
            #       1. PERFORM TESSELLATION ON THE REFERENCE ELEMENT -> OBTAIN NODAL COORDINATES OF REFERENCE SUBELEMENTS
            #       2. MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS
            #       3. MAP THE OBTAINED ADAPTED REFERENCE QUADRATURE TO PHYSICAL SPACE   
            #    - ADAPTED QUADRATURE TO INTEGRATE OVER INTERFACE
            #       1. MAP 1D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE APPROXIMATED INTERFACE
            #       2. MAP THE OBTAINED ADAPTED REFERENCE QUADRATURE TO PHYSICAL SPACE 
        """
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL SUBELEMENTS (2D)
        # TESSELLATE REFERENCE ELEMENT
        self.Nesub, SubElType, XIeTESSHO, self.interfedge = self.ReferenceElementTessellation()
        # MAP TESSELLATION TO PHYSICAL SPACE
        XeTESSHO = list()
        for isub in range(self.Nesub):
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS AT SUBELEMENTAL NODAL COORDINATES 
            N2D, foo, foo = EvaluateReferenceShapeFunctions(XIeTESSHO[isub], self.ElType, self.ElOrder)
            # MAP SUBELEMENTAL NODAL COORDINATES TO PHYSICAL SPACE
            XeTESSHO.append(N2D @ self.Xe)
            
        # LEVEL-SET NODAL VALUES
        if self.Dom == 0:  # ELEMENT CONTAINING PLASMA/VACUUM INTERFACE
            LSe = self.PlasmaLSe
        elif self.Dom == 2:  # ELEMENT CONTAINING VACUUM VESSEL FIRST WALL
            LSe = self.VacVessLSe
        
        # GENERATE SUBELEMENTAL OBJECTS
        self.SubElements = [Element(index = isubel, ElType = SubElType, ElOrder = self.ElOrder,
                                Xe = XeTESSHO[isubel],
                                Te = self.Te,
                                PlasmaLSe = None, 
                                VacVessLSe= None) for isubel in range(self.Nesub)]
        
        for isub, SUBELEM in enumerate(self.SubElements):
            #### ASSIGN REFERENCE SPACE TESSELLATION
            SUBELEM.XIe = XIeTESSHO[isub]
            #### ASSIGN A REGION FLAG TO EACH SUBELEMENT
            # INTERPOLATE VALUE OF LEVEL-SET FUNCTION INSIDE SUBELEMENT
            LSesub = self.ElementalInterpolationREFERENCE(np.mean(SUBELEM.XIe,axis=0),LSe)
            if self.Dom == 0:
                if LSesub < 0: 
                    SUBELEM.Dom = -1
                else:
                    SUBELEM.Dom = 1
            elif self.Dom == 2:
                if LSesub < 0: 
                    SUBELEM.Dom = 1
                else:
                    SUBELEM.Dom = 3
                    
        # COMPUTE INTEGRATION QUADRATURE FOR EACH SUBELEMENT
        for SUBELEM in self.SubElements:
            # STANDARD REFERENCE ELEMENT QUADRATURE (2D)
            XIg2Dstand, SUBELEM.Wg, SUBELEM.ng = GaussQuadrature(SUBELEM.ElType,NumQuadOrder)
            # EVALUATE SUBELEMENTAL REFERENCE SHAPE FUNCTIONS 
            N2Dstand, foo, foo = EvaluateReferenceShapeFunctions(XIg2Dstand, SUBELEM.ElType, SUBELEM.ElOrder)
            # MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS  ->> ADAPTED 2D QUADRATURE FOR SUBELEMENTS
            SUBELEM.XIg = N2Dstand @ SUBELEM.XIe
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS ON ADAPTED REFERENCE QUADRATURE
            SUBELEM.Ng, SUBELEM.dNdxig, SUBELEM.dNdetag = EvaluateReferenceShapeFunctions(SUBELEM.XIg, self.ElType, self.ElOrder)
            # MAPP ADAPTED REFERENCE QUADRATURE ON PHYSICAL ELEMENT
            SUBELEM.Xg = SUBELEM.Ng @ self.Xe
            
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON ADAPTED QUADRATURES NODES
            SUBELEM.invJg = np.zeros([SUBELEM.ng,SUBELEM.dim,SUBELEM.dim])
            SUBELEM.detJg = np.zeros([SUBELEM.ng])
            for ig in range(SUBELEM.ng):
                SUBELEM.invJg[ig,:,:], SUBELEM.detJg[ig] = Jacobian(self.Xe,SUBELEM.dNdxig[ig,:],SUBELEM.dNdetag[ig,:])
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL INTERFACE APPROXIMATION (1D)
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1Dstand, Wg1D, Ng1D = GaussQuadrature(0,NumQuadOrder)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.nedge-1)
                
        for INTERFACE in self.InterfApprox:
            for SEGMENT in INTERFACE.Segments:
                SEGMENT.ng = Ng1D
                SEGMENT.Wg = Wg1D
                SEGMENT.detJg = np.zeros([SEGMENT.ng])
                # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE ->> ADAPTED 1D QUADRATURE FOR INTERFACE
                SEGMENT.XIg = N1D @ SEGMENT.XIseg
                # EVALUATE 2D REFERENCE SHAPE FUNCTION ON INTERFACE ADAPTED QUADRATURE
                SEGMENT.Ng, SEGMENT.dNdxig, SEGMENT.dNdetag = EvaluateReferenceShapeFunctions(SEGMENT.XIg, self.ElType, self.ElOrder)
                # MAPP REFERENCE INTERFACE ADAPTED QUADRATURE ON PHYSICAL ELEMENT 
                SEGMENT.Xg = N1D @ SEGMENT.Xseg
                for ig in range(SEGMENT.ng):
                    SEGMENT.detJg[ig] = Jacobian1D(SEGMENT.Xseg,dNdxi1D[ig,:])  
        return 
    
    
    ##################################################################################################
    ################################ ELEMENTAL INTEGRATION ###########################################
    ##################################################################################################
    
    def IntegrateElementalDomainTerms(self,SourceTermg,*args):
        """ Input: - SourceTermg: source term (plasma current) evaluated at physical gauss integration nodes
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector
                    """
                    
        LHSe = np.zeros([len(self.Te),len(self.Te)])
        RHSe = np.zeros([len(self.Te)])
        
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.ng):  
            # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            R = self.Xg[ig,0]
            if args:   # DIMENSIONLESS SOLUTION CASE  -->> args[0] = R0
                Ngrad *= args[0]
                R /= args[0]
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
            for i in range(len(self.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(self.Te)):   # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) *(Jacobiano*2pi*rad) ]  
                    LHSe[i,j] -= Ngrad[:,j]@Ngrad[:,i]*self.detJg[ig]*self.Wg[ig]
                    ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j *(Jacobiano*2pi*rad) ]  ONLY RESPECT TO R
                    LHSe[i,j] += (1/R)*self.Ng[ig,j]*Ngrad[0,i]*self.detJg[ig]*self.Wg[ig]
                # COMPUTE RHS VECTOR TERMS [ (source term)*N_i*(Jacobiano *2pi*rad) ]
                RHSe[i] += SourceTermg[ig] * self.Ng[ig,i] *self.detJg[ig]*self.Wg[ig]
                
        return LHSe, RHSe
    
    
    def IntegrateElementalInterfaceTerms(self,beta,*args):
        """ Input: - PSIgseg: Interface condition, evaluated at physical gauss integration nodes
                   - beta: Nitsche's method penalty parameter
                   - LHS: global system Left-Hand-Side matrix 
                   - RHS: global system Reft-Hand-Side vector 
                    """
                    
        LHSe = np.zeros([len(self.Te),len(self.Te)])
        RHSe = np.zeros([len(self.Te)])
    
        # LOOP OVER SEGMENTS CONSTITUTING THE INTERFACE ELEMENTAL APPROXIMATION
        for INTERFACE in self.InterfApprox:
            for SEGMENT in INTERFACE.Segments:
                # LOOP OVER GAUSS INTEGRATION NODES
                for ig in range(SEGMENT.ng):  
                    # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
                    n_dot_Ngrad = SEGMENT.NormalVec@np.array([SEGMENT.dNdxig[ig,:],SEGMENT.dNdetag[ig,:]])
                    R = SEGMENT.Xg[ig,0]
                    if args:   # DIMENSIONLESS SOLUTION CASE  -->> args[0] = R0
                        n_dot_Ngrad *= args[0]
                        R /= args[0]
                    # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                    for i in range(len(self.Te)):  # ROWS ELEMENTAL MATRIX
                        for j in range(len(self.Te)):  # COLUMNS ELEMENTAL MATRIX
                            # COMPUTE LHS MATRIX TERMS
                            ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) *(Jacobiano*2pi*rad) ]  
                            LHSe[i,j] += SEGMENT.Ng[ig,i] * n_dot_Ngrad[j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                            ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) *(Jacobiano*2pi*rad) ]
                            LHSe[i,j] += n_dot_Ngrad[i]*SEGMENT.Ng[ig,j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                            ### PENALTY TERM   [ beta * (N_i*N_j) *(Jacobiano*2pi*rad) ]
                            LHSe[i,j] += beta * SEGMENT.Ng[ig,i] * SEGMENT.Ng[ig,j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                        # COMPUTE RHS VECTOR TERMS 
                        ### SYMMETRIC NITSCHE'S METHOD TERM  [ PSI_D * (n dot nabla(N_i)) * (Jacobiano *2pi*rad) ]
                        RHSe[i] +=  SEGMENT.PSIgseg[ig] * n_dot_Ngrad[i] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                        ### PENALTY TERM   [ beta * N_i * PSI_D *(Jacobiano*2pi*rad) ]
                        RHSe[i] +=  beta * SEGMENT.PSIgseg[ig] * SEGMENT.Ng[ig,i] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
        
        return LHSe, RHSe
    
    
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
    
    
def ReferenceElementCoordinates(elemType,elemOrder):
    match elemType:
        case 0:    # LINE (1D ELEMENT)
            match elemOrder:
                case 0:
                    # --1--
                    Xe = np.array([0])
                case 1:
                    # 1---2
                    Xe = np.array([-1,1])      
                case 2:         
                    # 1---3---2
                    Xe = np.array([-1,1,0])
                case 3:         
                    # 1-3-4-2
                    Xe = np.array([-1,1,-1/3,1/3])
    
        case 1:   # TRIANGLE
            match elemOrder:
                case 1:
                    # 2
                    # |\
                    # | \
                    # 3--1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0]])
                case 2:
                    # 2
                    # |\
                    # 5 4
                    # |  \
                    # 3-6-1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0],
                                   [1/2,1/2],
                                   [0,1/2],
                                   [1/2,0]])
                case 3:
                    #  2
                    # | \
                    # 6  5 
                    # |   \
                    # 7 10 4
                    # |     \
                    # 3-8--9-1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0],
                                   [2/3,1/3],
                                   [1/3,2/3],
                                   [0,2/3],
                                   [0,1/3],
                                   [1/3,0],
                                   [2/3,0],
                                   [1/3,1/3]])
                            
        case 2:    # QUADRILATERAL
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1]])
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1],
                                   [0,-1],
                                   [1,0],
                                   [0,1],
                                   [-1,0],
                                   [0,0]])
                case 3:
                    # 4---10--9---3
                    # |           |
                    # 11  16  15  8
                    # |           |
                    # 12  13  14  7
                    # |           |
                    # 1---5---6---2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1],
                                   [-1/3,-1],
                                   [1/3,-1],
                                   [1,-1/3],
                                   [1,1/3],
                                   [1/3,1],
                                   [-1/3,1],
                                   [-1,1/3],
                                   [-1,-1/3],
                                   [-1/3,-1/3],
                                   [1/3,-1/3],
                                   [1/3,1/3],
                                   [-1/3,1/3]])
    return Xe
