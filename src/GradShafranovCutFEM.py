""" This script contains the Python object defining a plasma equilibrium problem, modeled using the Grad-Shafranov equation
in an axisymmetrical system such as a tokamak. """

import numpy as np
import matplotlib.pyplot as plt
from random import random
from scipy.interpolate import griddata
from src.GaussQuadrature import *
from src.ShapeFunctions import *
from src.ElementObject import *

class GradShafranovCutFEM:
    
    # GENERAL PARAMETERS
    epsilon0 = 8.8542E-12       # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19               # J eV-1   Botlzmann constant

    def __init__(self,mesh_folder,EQU_case_file,ElementType,ElementOrder,QuadratureOrder):
        # INPUT FILES:
        self.mesh_folder = mesh_folder
        self.MESH = mesh_folder[mesh_folder.rfind("/")+1:]
        self.case_file = EQU_case_file
        
        # DECLARE PROBLEM ATTRIBUTES
        self.PLASMA_BOUNDARY = None         # PLASMA BOUNDARY BEHAVIOUR: 'FIXED'  or  'FREE'
        self.CASE = None                    # CASE SOLUTION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.InterElems = None              # LIST OF CUT ELEMENTS (INDEXES), CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.BoundaryElems = None           # LIST OF ELEMENTS (INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.LevelSet = None                # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.PHI = None                     # PHI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.PHI_0 = None                   # PHI VALUE AT MAGNETIC AXIS MINIMA
        self.PHI_X = None                   # PHI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PHI_NORM = None                # NORMALISED PHI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PHI_B = None                   # VACUUM VESSEL BOUNDARY PHI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PHI_CONV = None                # CONVERGED NORMALISED PHI SOLUTION FIELD  
        
        # PARAMETERS FOR COILS
        self.Ncoils = None                  # TOTAL NUMBER OF COILS
        self.Xcoils = None                  # COILS' COORDINATE MATRIX 
        self.Icoils = None                  # COILS' CURRENT
        # PARAMETERS FOR SOLENOIDS
        self.Nsolenoids = None              # TOTAL NUMBER OF SOLENOIDS
        self.Xsolenoids = None              # SOLENOIDS' COORDINATE MATRIX
        self.Nturnssole = None              # SOLENOIDS' NUMBER OF TURNS
        self.Isolenoids = None              # SOLENOIDS' CURRENT
        # PLASMA REGION GEOMETRY
        self.epsilon = None                 # PLASMA REGION ASPECT RATIO
        self.kappa = None                   # PLASMA REGION ELONGATION
        self.delta = None                   # PLASMA REGION TRIANGULARITY
        self.Rmax = None                    # PLASMA REGION MAJOR RADIUS
        self.Rmin = None                    # PLASMA REGION MINOR RADIUS
        self.R0 = None                      # PLASMA REGION MEAN RADIUS
        
        # COMPUTATIONAL MESH
        self.ElType = ElementType           # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = ElementOrder         # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.nedge = None                   # NUMBER OF NODES ON ELEMENT EDGE
        self.dim = None                     # SPACE DIMENSION
        self.Tbound = None                  # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        self.Nbound = None                  # NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
        self.Nnbound = None                 # NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodes = None           # LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.Xmax = None                    # COMPUTATIONAL MESH MAXIMAL X (R) COORDINATE
        self.Xmin = None                    # COMPUTATIONAL MESH MINIMAL X (R) COORDINATE
        self.Ymax = None                    # COMPUTATIONAL MESH MAXIMAL Y (Z) COORDINATE
        self.Ymin = None                    # COMPUTATIONAL MESH MINIMAL Y (Z) COORDINATE
        
        # NUMERICAL TREATMENT PARAMETERS
        self.QuadratureOrder = QuadratureOrder   # NUMERICAL INTEGRATION QUADRATURE ORDER
        #### 1D NUMERICAL INTEGRATION QUADRATURE TO INTEGRATE ALONG SOLENOIDS 
        self.Ng1D = None                         # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 1D QUADRATURE
        self.XIg1D = None                        # STANDARD 1D GAUSS INTEGRATION NODES 
        self.Wg1D = None                         # STANDARD 1D GAUSS INTEGRATION WEIGTHS 
        self.N1D = None                          # REFERENCE 1D SHAPE FUNCTIONS EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES 
        self.dNdxi1D = None                      # REFERENCE 1D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT STANDARD 1D GAUSS INTEGRATION NODES
        #### DOBLE WHILE LOOP STRUCTURE PARAMETERS
        self.INT_TOL = None                      # INTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.EXT_TOL = None                      # EXTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.INT_ITER = None                     # INTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.EXT_ITER = None                     # EXTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.converg_EXT = None                  # EXTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.converg_INT = None                  # INTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.it_EXT = None                       # EXTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it_INT = None                       # INTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it = 0                              # TOTAL NUMBER OF ITERATIONS COUNTER
        self.alpha = None                        # AIKTEN'S SCHEME RELAXATION CONSTANT
        #### BOUNDARY CONSTRAINTS
        self.beta = 1e8                          # NITSCHE'S METHOD PENALTY TERM
        self.coeffs = []                         # ANALYTICAL SOLUTION/INITIAL GUESS COEFFICIENTS
        self.nsole = 2                           # NUMBER OF NODES ON SOLENOID ELEMENT
        
        return
    
    def print_all_attributes(self):
        """ Function which prints all object EQUILI attributes and their corresponding values. """
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")
        return
    
    
    ##################################################################################################
    ############################### READ INPUT DATA FILES ############################################
    ##################################################################################################
    
    def ReadMesh(self):
        """ Reads input mesh data files. """
        
        print("     -> READ MESH DATA FILES...",end='')
        # NUMBER OF NODES PER ELEMENT
        self.n, self.nedge = ElementalNumberOfNodes(self.ElType, self.ElOrder)
        
        # READ DOM FILE .dom.dat
        MeshDataFile = self.mesh_folder +'/'+ self.MESH +'.dom.dat'
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
        MeshFile = self.mesh_folder +'/'+ self.MESH +'.geo.dat'
        self.T = np.zeros([self.Ne,self.n], dtype = int)
        self.X = np.zeros([self.Nn,self.dim], dtype = float)
        self.Tbound = np.zeros([self.Nbound,self.nedge+1], dtype = int)   # LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE 
        file = open(MeshFile, 'r') 
        i = -1
        j = -1
        k = -1
        for line in file:
            # first we format the line read in order to remove all the '\n'  
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA STARTS
            if l[0] == 'ELEMENTS':
                i=0
                continue
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA ENDS
            elif l[0] == 'END_ELEMENTS':
                i=-1
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA STARTS
            elif l[0] == 'COORDINATES':
                j=0
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA ENDS
            elif l[0] == 'END_COORDINATES':
                j=-1
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA STARTS
            elif l[0] == 'BOUNDARIES,':
                k=0
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA ENDS
            elif l[0] == 'END_BOUNDARIES':
                k=-1
                continue
            if i>=0:
                for m in range(self.n):
                    self.T[i,m] = int(l[m+1])
                i += 1
            if j>=0:
                for m in range(self.dim):
                    self.X[j,m] = float(l[m+1])
                j += 1
            if k>=0:
                for m in range(self.nedge+1):
                    self.Tbound[k,m] = float(l[m+1])
                k += 1
        file.close()
        # PYTHON INDEXES START AT 0 AND NOT AT 1. THUS, THE CONNECTIVITY MATRIX INDEXES MUST BE MODIFIED
        self.T = self.T - 1
        self.Tbound = self.Tbound - 1
        
        # OBTAIN BOUNDARY NODES
        self.BoundaryNodes = set()     # GLOBAL INDEXES OF NODES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        for i in range(self.nedge):
            for node in self.Tbound[:,i]:
                self.BoundaryNodes.add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodes = np.array(sorted(self.BoundaryNodes))
        self.Nnbound = len(self.BoundaryNodes)
        
        # OBTAIN COMPUTATIONAL MESH LIMITS
        self.Xmax = np.max(self.X[:,0])
        self.Xmin = np.min(self.X[:,0])
        self.Ymax = np.max(self.X[:,1])
        self.Ymin = np.min(self.X[:,1])
        
        print('Done!')
        return
    
    def ReadEQUILIdata(self):
        """ Reads problem data from input file equ.dat. That is:
                - SOLUTION CASE AND PROBLEM TYPE (FIXED/FREE BOUNDARY)
                - TOTAL CURRENT
                - PLASMA GEOMETRY DATA
                - LOCATION AND CURRENT OF EXTERNAL COILS CONFINING THE PLASMA
                - PLASMA PROPERTIES
                - NUMERICAL TREATMENT
                """
                
        print("     -> READ EQUILI DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        EQUILIDataFile = self.case_file +'.equ.dat'
        file = open(EQUILIDataFile, 'r') 
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
                    
            if l:  # LINE NOT EMPTY
                if l[0] == 'PLASMA_BOUNDARY:':      # READ IF FIXED/FREE BOUNDARY PROBLEM
                    self.PLASMA_BOUNDARY = l[1]
                if l[0] == 'SOL_CASE:':             # READ SOLUTION CASE
                    self.CASE = l[1]
                if l[0] == 'TOTAL_CURRENT:':        # READ TOTAL PLASMA CURRENT
                    self.TOTAL_CURRENT = float(l[1])
                    
                ### IF FIXED-BOUNDARY PROBLEM:
                if self.PLASMA_BOUNDARY == "FIXED":
                    # READ PLASMA GEOMETRY PARAMETERS
                    if l[0] == 'R_MAX:':    # READ PLASMA REGION X_CENTER 
                        self.Rmax = float(l[1])
                    elif l[0] == 'R_MIN:':    # READ PLASMA REGION Y_CENTER 
                        self.Rmin = float(l[1])
                    elif l[0] == 'EPSILON:':    # READ PLASMA REGION X_MINOR 
                        self.epsilon = float(l[1])
                    elif l[0] == 'KAPPA:':    # READ PLASMA REGION X_MAYOR 
                        self.kappa = float(l[1])
                    elif l[0] == 'DELTA:':    # READ PLASMA REGION X_CENTER 
                        self.delta = float(l[1])
                    
                ### IF FREE-BOUNDARY PROBLEM:
                elif self.PLASMA_BOUNDARY == 'FREE':
                    # READ PLASMA SHAPE CONTROL POINTS
                    if l[0] == 'X_SADDLE:':    # READ PLASMA REGION X_CENTER 
                        self.X_SADDLE = float(l[1])
                    elif l[0] == 'Y_SADDLE:':    # READ PLASMA REGION Y_CENTER 
                        self.Y_SADDLE = float(l[1])
                    elif l[0] == 'X_RIGHTMOST:':    # READ PLASMA REGION X_CENTER 
                        self.X_RIGHTMOST = float(l[1])
                    elif l[0] == 'Y_RIGHTMOST:':    # READ PLASMA REGION Y_CENTER 
                        self.Y_RIGHTMOST = float(l[1])
                    elif l[0] == 'X_LEFTMOST:':    # READ PLASMA REGION X_CENTER 
                        self.X_LEFTMOST = float(l[1])
                    elif l[0] == 'Y_LEFTMOST:':    # READ PLASMA REGION Y_CENTER 
                        self.Y_LEFTMOST = float(l[1])
                    elif l[0] == 'X_TOP:':    # READ PLASMA REGION X_CENTER 
                        self.X_TOP = float(l[1])
                    elif l[0] == 'Y_TOP:':    # READ PLASMA REGION Y_CENTER 
                        self.Y_TOP = float(l[1])
                    
                    
                    # READ PLASMA GEOMETRY PARAMETERS
                    if l[0] == 'X_CENTRE:':    # READ PLASMA REGION X_CENTER 
                        self.X_CENTRE = float(l[1])
                    elif l[0] == 'Y_CENTRE:':    # READ PLASMA REGION Y_CENTER 
                        self.Y_CENTRE = float(l[1])
                    elif l[0] == 'X_MINOR:':    # READ PLASMA REGION X_MINOR 
                        self.X_MINOR = float(l[1])
                    elif l[0] == 'X_MAYOR:':    # READ PLASMA REGION X_MAYOR 
                        self.X_MAYOR = float(l[1])
                    elif l[0] == 'YUP_MAYOR':    # READ PLASMA REGION X_CENTER 
                        self.YUP_MAYOR = float(l[1])
                    elif l[0] == 'XYUP_MAYOR':    # READ PLASMA REGION X_CENTER 
                        self.XYUP_MAYOR = float(l[1])
                    elif l[0] == 'YDO_MAYOR':    # READ PLASMA REGION X_CENTER 
                        self.YDO_MAYOR = float(l[1])
                    elif l[0] == 'XYDO_MAYOR':    # READ PLASMA REGION X_CENTER 
                        self.XYDO_MAYOR = float(l[1])
                        
                    # READ COIL PARAMETERS
                    elif l[0] == 'N_COILS:':    # READ PLASMA REGION X_CENTER 
                        self.Ncoils = int(l[1])
                        self.Xcoils = np.zeros([self.Ncoils,self.dim])
                        self.Icoils = np.zeros([self.Ncoils])
                        i = 0
                    elif l[0] == 'Xposi:' and i<self.Ncoils:    # READ i-th COIL X POSITION
                        self.Xcoils[i,0] = float(l[1])
                    elif l[0] == 'Yposi:' and i<self.Ncoils:    # READ i-th COIL Y POSITION
                        self.Xcoils[i,1] = float(l[1])
                    elif l[0] == 'Inten:' and i<self.Ncoils:    # READ i-th COIL INTENSITY
                        self.Icoils[i] = float(l[1])
                        i += 1
                        
                    # READ SOLENOID PARAMETERS:
                    elif l[0] == 'N_SOLENOIDS:':    # READ PLASMA REGION X_CENTER 
                        self.Nsolenoids = int(l[1])
                        self.Xsolenoids = np.zeros([self.Nsolenoids,self.dim+1])
                        self.Nturnssole = np.zeros([self.Nsolenoids])
                        self.Isolenoids = np.zeros([self.Nsolenoids])
                        j = 0
                    elif l[0] == 'Xposi:' and j<self.Nsolenoids:    # READ j-th SOLENOID X POSITION
                        self.Xsolenoids[j,0] = float(l[1])
                    elif l[0] == 'Ylow:' and j<self.Nsolenoids:    # READ j-th SOLENOID Y POSITION
                        self.Xsolenoids[j,1] = float(l[1])
                    elif l[0] == 'Yup:' and j<self.Nsolenoids:    # READ j-th SOLENOID Y POSITION
                        self.Xsolenoids[j,2] = float(l[1])
                    elif l[0] == 'Turns:' and j<self.Nsolenoids:    # READ j-th SOLENOID NUMBER OF TURNS
                        self.Nturnssole[j] = float(l[1])
                    elif l[0] == 'Inten:' and j<self.Nsolenoids:    # READ j-th SOLENOID INTENSITY
                        self.Isolenoids[j] = float(l[1])
                        j += 1
                    
                # READ NUMERICAL TREATMENT PARAMETERS
                if l[0] == 'EXT_ITER:':    # READ MAXIMAL NUMBER OF ITERATION FOR EXTERNAL LOOP
                    self.EXT_ITER = int(l[1])
                elif l[0] == 'EXT_TOL:':    # READ TOLERANCE FOR EXTERNAL LOOP
                    self.EXT_TOL = float(l[1])
                elif l[0] == 'INT_ITER:':    # READ MAXIMAL NUMBER OF ITERATION FOR INTERNAL LOOP
                    self.INT_ITER = int(l[1])
                elif l[0] == 'INT_TOL:':    # READ TOLERANCE FOR INTERNAL LOOP
                    self.INT_TOL = float(l[1])
                elif l[0] == 'RELAXATION:':   # READ AITKEN'S SCHEME RELAXATION CONSTANT
                    self.alpha = float(l[1])
                    
        if self.PLASMA_BOUNDARY == "FIXED":
            self.R0 = (self.Rmax+self.Rmin)/2      
            
        print('Done!')  
        return
    
    
    ##################################################################################################
    ############################# INITIAL GUESS AND SOLUTION CASE ####################################
    ##################################################################################################
    
    def ComputeLinearSolutionCoefficients(self):
        """ Computes the coeffients for the magnetic flux in the linear source term case, that is for 
                    GRAD-SHAFRANOV EQ:  DELTA*(PHI) = R^2   (plasma current is linear such that Jphi = R/mu0)
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
        return coeffs.T[0].tolist() 
    
    def SolutionCASE(self,X):
        """ Function which computes the analytical solution (if it exists) at point with coordinates X. """
        
        if self.PLASMA_BOUNDARY == 'FIXED':
            # DIMENSIONALESS COORDINATES
            Xstar = X/self.R0
            if self.CASE == 'LINEAR':
                if not self.coeffs: 
                    self.coeffs = self.ComputeLinearSolutionCoefficients()  # [D1, D2, D3]
                PHIexact = (Xstar[0]**4)/8 + self.coeffs[0] + self.coeffs[1]*Xstar[0]**2 + self.coeffs[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
                
            elif self.CASE == 'NONLINEAR':
                if not self.coeffs:
                    self.coeffs = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0]
                PHIexact = np.sin(self.coeffs[0]*(Xstar[0]+self.coeffs[2]))*np.cos(self.coeffs[1]*Xstar[1])
            
        elif self.PLASMA_BOUNDARY == 'FREE': 
            if self.CASE == 'CASE0':
                PHIexact = 0 
            
        return PHIexact
    
    
    ##################################################################################################
    ###################################### PLASMA CURRENT ############################################
    ##################################################################################################
    
    def Jphi(self,R,Z,phi):
        """ Function which computes the plasma current source term on the right hand side of the Grad-Shafranov equation. """
        
        if self.PLASMA_BOUNDARY == 'FIXED':
            # NORMALIE COORDINATES
            R = R/self.R0
            Z = Z/self.R0
            # COMPUTE  PLASMA CURRENT
            if self.CASE == 'LINEAR':
                # self.coeffs = [D1 D2 D3]  for linear solution
                jphi = R/self.mu0
                
            if self.CASE == 'NONLINEAR': 
                # self.coeffs = [Kr Kz R0]  for nonlinear solution
                jphi = -((self.coeffs[0]**2+self.coeffs[1]**2)*phi+(self.coeffs[0]/R)*np.cos(self.coeffs[0]*(R+self.coeffs[2]))*np.cos(self.coeffs[1]*Z)
                +R*((np.sin(self.coeffs[0]*(R+self.coeffs[2]))*np.cos(self.coeffs[1]*Z))**2-phi**2+np.exp(-np.sin(self.coeffs[0]*(R+self.coeffs[2]))*
                                                                                            np.cos(self.coeffs[1]*Z))-np.exp(-phi)))/(self.mu0*R)
        return jphi
    
    
    ##################################################################################################
    ###################################### LEVEL-SET DESCRIPTION #####################################
    ##################################################################################################
    
    def F4E_LevelSet(self):
        """ # IN ORDER TO FIND THE CURVE PARAMETRIZING THE PLASMA REGION BOUNDARY, WE LOOK FOR THE COEFFICIENTS DEFINING
        # A 3rd ORDER HAMILTONIAN FROM WHICH WE WILL TAKE THE 0-LEVEL CURVE AS PLASMA REGION BOUNDARY. THAT IS
        #
        # H(x,y) = A00 + A10x + A01y + A20x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # HENCE, WE NEED TO IMPOSE CONSTRAINTS ON THE HAMILTONIAN FUNCTION IN ORDER TO SOLVE A SYSTEM OF EQUATIONS 
        # (LINEAR OR NONLINEAR). THE RESULTING SYSTEM WILL READ AS   Ax = b.
        # IN ORDER TO SIMPLIFY SUCH PROBLEM, WE ASSUME THAT:
        #   - ORIGIN (0,0) ON 0-LEVEL CURVE ---> A00 = 0
        #   - SADDLE POINT AT (0,0) ---> A10 = A01 = 0 
        # EVEN IF THAT IS NOT THE CASE IN THE PHYSICAL PLASMA REGION, WE ONLY NEED TO TRANSLATE THE REFERENCE FRAME 
        # RESPECT TO THE REAL SADDLE POINT LOCATION P0 IN ORDER TO WORK WITH EQUIVALENT PROBLEMS.
        # FURTHERMORE, WE CAN NORMALISE RESPECT TO A20 WITHOUT LOSS OF GENERALITY. THEREFORE, WE DEPART FROM 
        #
        # H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # AS MENTIONED EARLIER, THE PROFILE WILL CORRESPOND TO THE 0-LEVEL CURVE, WHICH MEANS WE MUST OBTAIN THE 
        # COEFFICIENTS FOR 
        #
        # A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3 = -x**2
        #
        # WE NEED HENCE TO IMPOSE 6 CONSTRAINTS IN ORDER TO DETERMINE THE REMAINING COEFFICIENTS
        
        # For this method we constraint the curve to:
        # - go through points P1, P2 and P3 (CONTROL POINTS)
        # - have vertical tangents at points P1 and P2
        # - have a 90º angle at saddle point
        
        # where the control points are defined as:
        #      - P0: SADDLE POINT
        #      - P1: RIGHTMOST POINT
        #      - P2: LEFTMOST POINT
        #      - P3: TOP POINT
        
        # Input: - P0: SADDLE POINT COORDINATES
        #        - P1: RIGHTMOST POINT COORDINATES
        #        - P2: LEFTMOST POINT COORDINATES
        #        - P3: TOP POINT COORDINATES
        #        - X: NODAL COORDINATES MATRIX
        # """
        
        # THE FOLLOWING FUNCTIONS TRANSLATE THE CONSTRAINTS ON THE PROBLEM INTO EQUATIONS FOR THE FINAL SYSTEM OF EQUATIONS TO SOLVE
        def Point_on_curve(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation 
            # obtained when constraining the curve to pass through point P. Such equation corresponds 
            # basically to   H(P) = 0.
            x, y = P
            Arow = [x*y, y**2, x**3, x**2*y, x*y**2, y**3]
            brow = -x**2
            return Arow, brow

        def VerticalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a vertical tangent at point P. Such equation  
            # corresponds basically to   dH/dy(P) = 0.
            x, y = P
            Arow = [x, 2*y, 0, x**2, 2*x*y, 3*y**2]
            brow = 0
            return Arow, brow

        def HorizontalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a horizontal tangent at point P. Such equation  
            # corresponds basically to   dH/dx(P) = 0.
            x, y = P
            Arow = [y, 0, 3*x**2, 2*x*y, y**2, 0]
            brow = -2*x
            return Arow, brow

        def RightAngle_SaddlePoint(A,b):
            # Function imposing a 90º angle at the closed surface saddle point at (0,0), which can be shown 
            # is equivalent to fixing  A02 = -1
            # Hence, what we need to do is take the second column of matrix A, corresponding to the A02 factors,
            # multiply them by -1 and pass them to the system's RHS, vector b. Then, we will reduce the system size.
            
            bred = np.delete(b+A[:,1].reshape((6,1)),5,0)     # pass second column to RHS and delete last row
            A = np.delete(A,1,1)    # delete second column 
            Ared = np.delete(A,5,0)    # delete last row
            return Ared, bred
        
        # BUILD CONTROL POINTS
        P0 = np.array([self.X_SADDLE, self.Y_SADDLE])
        P1 = np.array([self.X_RIGHTMOST, self.Y_RIGHTMOST])
        P2 = np.array([self.X_LEFTMOST, self.Y_LEFTMOST])
        P3 = np.array([self.X_TOP, self.Y_TOP])
        
        # 1. RESCALE POINT COORDINATES SO THAT THE SADDLE POINT IS LOCATED AT ORIGIN (0,0)
        P1star = P1-P0
        P2star = P2-P0
        P3star = P3-P0

        # 2. COMPUTE HAMILTONIAN COEFFICIENTS
        # Build system matrices
        A = np.zeros([6,6])
        b = np.zeros([6,1])

        # Constraints on point P1 = (a1,b1)
        Arow11, brow11 = Point_on_curve(P1star)
        Arow12, brow12 = VerticalTangent(P1star)
        A[0,:] = Arow11
        b[0] = brow11
        A[1,:] = Arow12
        b[1] = brow12

        # Constraints on point P2 = (a2,b2)
        Arow21, brow21 = Point_on_curve(P2star)
        Arow22, brow22 = VerticalTangent(P2star)
        A[2,:] = Arow21
        b[2] = brow21
        A[3,:] = Arow22
        b[3] = brow22
        
        # Constraints on point P3 = (a3,b3)
        Arow31, brow31 = Point_on_curve(P3star)
        A[4,:] = Arow31
        b[4] = brow31

        # 90º on saddle point (0,0)
        Ared, bred = RightAngle_SaddlePoint(A,b)   # Now A = [5x5] and  b = [5x1]
        
        # Solve system of equations and obtain Hamiltonian coefficients
        Q, R = np.linalg.qr(Ared)
        y = np.dot(Q.T, bred)
        coeffs_red = np.linalg.solve(R, y)  # Hamiltonian coefficients  [5x1]
        
        coeffs = np.insert(coeffs_red,1,-1,0)        # insert second coefficient A02 = -1
        
        # 2. OBTAIN LEVEL-SET VALUES ON MESH NODES
        # HAMILTONIAN  ->>  Z(x,y) = H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        Xstar = self.X[:,0]-P0[0]
        Ystar = self.X[:,1]-P0[1]
        LS = np.zeros([self.Nn])
        for i in range(self.Nn):
            LS[i] = Xstar[i]**2+coeffs[0]*Xstar[i]*Ystar[i]+coeffs[1]*Ystar[i]**2+coeffs[2]*Xstar[i]**3+coeffs[3]*Xstar[i]**2*Ystar[i]+coeffs[4]*Xstar[i]*Ystar[i]**2+coeffs[5]*Ystar[i]**3

        # MODIFY HAMILTONIAN VALUES SO THAT OUTSIDE THE PLASMA REGION THE LEVEL-SET IS POSITIVE  
        for i in range(self.Nn):
                if self.X[i,0] < P2[0] or self.X[i,1] < P0[1]:
                    LS[i] = np.abs(LS[i])
        return LS
    
    
    
    ##################################################################################################
    ################################# VACUUM VESSEL BOUNDARY PHI_B ###################################
    ##################################################################################################
    
    def ComputePHI_B(self,STEP):
        
        """ FUNCTION TO COMPUTE THE COMPUTATIONAL DOMAIN BOUNDARY VALUES FOR PHI, PHI_B, ON BOUNDARY ELEMENT. 
        THESE MUST BE TREATED AS NATURAL BOUNDARY CONDITIONS (DIRICHLET BOUNDARY CONDITIONS).
        FOR 'FREE' BOUNDARY PROBLEM, SUCH VALUES ARE OBTAINED BY ACCOUNTING FOR THE CONTRIBUTIONS FROM THE EXTERNAL
        FIXED COILS AND THE CONTRIBUTION FROM THE PLASMA CURRENT ITSELF, FOR WHICH WE 
        INTEGRATE THE PLASMA'S GREEN FUNCTION."""
        
        def ellipticK(k):
            """ COMPLETE ELLIPTIC INTEGRAL OF 1rst KIND """
            pk=1.0-k*k
            if k == 1:
                ellipticK=1.0e+16
            else:
                AK = (((0.01451196212*pk+0.03742563713)*pk +0.03590092383)*pk+0.09666344259)*pk+1.38629436112
                BK = (((0.00441787012*pk+0.03328355346)*pk+0.06880248576)*pk+0.12498593597)*pk+0.5
                ellipticK = AK-BK*np.log(pk)
            return ellipticK

        def ellipticE(k):
            """COMPLETE ELLIPTIC INTEGRAL OF 2nd KIND"""
            pk = 1 - k*k
            if k == 1:
                ellipticE = 1
            else:
                AE=(((0.01736506451*pk+0.04757383546)*pk+0.0626060122)*pk+0.44325141463)*pk+1
                BE=(((0.00526449639*pk+0.04069697526)*pk+0.09200180037)*pk+0.2499836831)*pk
                ellipticE = AE-BE*np.log(pk)
            return ellipticE
        
        def GreenFunction(Xb,Xp):
            """ GREEN FUNCTION CORRESPONDING TO THE TOROIDAL ELLIPTIC OPERATOR """
            kcte= np.sqrt(4*Xb[0]*Xp[0]/((Xb[0]+Xp[0])**2 + (Xp[1]-Xb[1])**2))
            Greenfun = (1/(2*np.pi))*(np.sqrt(Xp[0]*Xb[0])/kcte)*((2-kcte**2)*ellipticK(kcte)-2*ellipticE(kcte))
            return Greenfun
        
        """ RELEVANT ATTRIBUTES:
                # Nbound: NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
                # Nnbound: NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
                # BoundaryNodes: LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
                # Tbound: MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
                # Ncoils: TOTAL NUMBER OF COILS
                # Xcoils: COILS' COORDINATE MATRIX 
                # Icoils: COILS' CURRENT
                # Nsolenoids: TOTAL NUMBER OF SOLENOIDS
                # Xsolenoids: SOLENOIDS' COORDINATE MATRIX
                # Nturnssole: SOLENOIDS' NUMBER OF TURNS
                # Isolenoids: SOLENOIDS' CURRENT"""
                
        if self.PLASMA_BOUNDARY == 'FIXED':  # FOR FIXED PLASMA BOUNDARY PROBLEM THE VACUUM VESSEL BOUNDARY VALUES PHI_B ARE IRRELEVANT ->> PHI_B = 0
            return
        
        elif self.PLASMA_BOUNDARY == 'FREE':
            if STEP == 'INITIALIZATION':
                column = 0    # STORE VALUES IN FIRST COLUMN (ITERATION N)
            elif STEP == 'ITERATION':
                column = 1    # STORE VALUES IN SECOND COLUMN
            
            # COMPUTE PHI_B VALUE ON EACH VACUUM VESSEL BOUNDARY NODE (ITERATION N+1)
            for i in range(self.Nnbound):
                # ISOLATE NODAL COORDINATES
                Xnode = self.X[self.BoundaryNodes[i],:]
                
                # CONTRIBUTION FROM EXTERNAL COILS CURRENT 
                for icoil in range(self.Ncoils): 
                    self.PHI_B[i,column] += self.Icoils[icoil] * GreenFunction(Xnode,self.Xcoils[icoil,:]) 
                
                # CONTRIBUTION FROM EXTERNAL SOLENOIDS CURRENT  ->>  INTEGRATE OVER SOLENOID LENGTH 
                for isole in range(self.Nsole):
                    Xsolenoid = np.array([[self.Xsole[isole,0], self.Xsole[isole,1]],[self.Xsole[isole,0], self.Xsole[isole,2]]])   # COORDINATES OF SOLENOID EDGES
                    Jsole = self.Isole[isole]*self.Nturnsole[isole]/np.linalg.norm(Xsolenoid[0,:]-Xsolenoid[1,:])   # SOLENOID CURRENT LINEAR DENSITY
                    # LOOP OVER GAUSS NODES
                    for ig in range(self.Ng1D):
                        # MAPP 1D REFERENCE INTEGRATION GAUSS NODES TO PHYSICAL SPACE ON SOLENOID
                        Xgsole = self.N1D[ig,:] @ Xsolenoid
                        # COMPUTE DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID 
                        detJ1D = Jacobian1D(Xsolenoid[0,:],Xsolenoid[1,:],self.dNdxi1D[ig,:])
                        detJ1D = detJ1D*2*np.pi*self.Xsole[isole,0]
                        for k in range(self.nsole):
                            self.PHI_B[i,column] += GreenFunction(Xnode,Xgsole) * Jsole * self.N1D[ig,k] * detJ1D * self.Wg1D[ig]
                    
                # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                #   1. INTEGRATE IN PLASMA ELEMENTS
                for elem in self.PlasmaElements:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.Elements[elem]
                    # INTERPOLATE ELEMENTAL PHI ON PHYSICAL GAUSS NODES
                    PHIg = ELEMENT.N @ ELEMENT.PHIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(ELEMENT.Ng2D):
                        for k in range(ELEMENT.n):
                            self.PHI_B[i,column] += GreenFunction(Xnode, ELEMENT.Xg2D[ig,:])*self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],
                                                                    PHIg[ig])*ELEMENT.N[ig,k]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]
                            
                #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                for elem in self.InterElements:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.Elements[elem]
                    # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                    for SUBELEM in ELEMENT.SubElements:
                        if SUBELEM.Dom < 0:  # IN PLASMA REGION
                            # INTERPOLATE ELEMENTAL PHI ON PHYSICAL GAUSS NODES
                            PHIg = SUBELEM.N @ ELEMENT.PHIe
                            # LOOP OVER GAUSS NODES
                            for ig in range(SUBELEM.Ng2D):
                                for k in range(SUBELEM.n):
                                    self.PHI_B[i,column] += GreenFunction(Xnode, SUBELEM.Xg2D[ig,:])*self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],
                                                                    PHIg[ig])*SUBELEM.N[ig,k]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]                   
            return
    
    ##################################################################################################
    ###################################### ELEMENTS DEFINITION #######################################
    ##################################################################################################
    
    def ClassifyElements(self):
        """ Function that sperates the elements into 4 groups: 
                - PlasmaElems: elements inside the plasma region P(phi) where the plasma current is different from 0
                - VacuumElems: elements outside the plasma region P(phi) where the plasma current is 0
                - InterElems: elements containing the plasma region's interface 
                - BoundaryElems: elements located at the computational domain's boundary, outside the plasma region P(phi) where the plasma current is 0. """
        
        self.PlasmaElems = np.zeros([self.Ne], dtype=int)    # GLOBAL INDEXES OF ELEMENTS INSIDE PLASMA REGION
        self.VacuumElems = np.zeros([self.Ne], dtype=int)    # GLOBAL INDEXES OF ELEMENTS OUTSIDE PLASMA REGION (VACUUM REGION)
        self.InterElems = np.zeros([self.Ne], dtype=int)     # GLOBAL INDEXES OF CUT ELEMENTS, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM 
        self.BoundaryElems = np.zeros([self.Ne], dtype=int)  # GLOBAL INDEXES OF ELEMENTS ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        kplasm = 0
        kvacuu = 0
        kint = 0
        kbound = 0
        
        for e in range(self.Ne):
            # WE FIRST EXTRACT THE LIST OF BOUNDARY ELEMENTS THANKS TO THE CONNECTIVITY MATRIX Tbound OBTAINED FROM THE MESH INPUT FILE DATA
            if e in self.Tbound[:,-1]:
                self.BoundaryElems[kbound] = e
                self.Elements[e].Dom = +1
                kbound += 1
            else:
                LSe = self.Elements[e].LSe  # elemental nodal level-set values
                for i in range(self.n-1):
                    if np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between nodal values change -> interface element
                        self.InterElems[kint] = e
                        self.Elements[e].Dom = 0
                        kint += 1
                        break
                    else:
                        if i+2 == self.n:   # if all nodal values have the same sign
                            if np.sign(LSe[i+1]) > 0:   # all nodal values with positive sign -> vacuum vessel element
                                self.Elements[e].Dom = +1
                                self.VacuumElems[kvacuu] = e
                                kvacuu += 1
                            else:   # all nodal values with negative sign -> plasma region element 
                                self.PlasmaElems[kplasm] = e
                                self.Elements[e].Dom = -1
                                kplasm += 1
                            
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.InterElems = self.InterElems[:kint]
        self.BoundaryElems = self.BoundaryElems[:kbound]
    
        return
    
    
    ##################################################################################################
    ############################### SOLUTION NORMALISATION ###########################################
    ##################################################################################################
    
    def ComputeCriticalPHI(self):
        """ Function which computes the values of PHI at the:
                - MAGNETIC AXIS ->> PHI_0 
                - SEPARATRIX (LAST CLOSED MAGNETIC SURFACE) / SADDLE POINT ->> PHI_X 
        These values are used to NORMALISE PHI. 
        
        THE METHODOLOGY IS THE FOLLOWING:
            1. OBTAIN CANDIDATE POINTS FOR SOLUTIONS OF EQUATION     NORM(GRAD(PHI))^2 = 0
            2. USING A NEWTON METHOD (OR SOLVER), FIND SOLUTION OF    NORM(GRAD(PHI))^2 = 0
            3. CHECK HESSIAN AT SOLUTIONS TO DIFFERENTIATE BETWEEN EXTREMUM AND SADDLE POINT
            
        THIS IS WHAT WE WOULD DO ANALYTICALLY. IN THE NUMERICAL CASE, WE DO:
            1. INTERPOLATE PHI VALUES ON A FINER STRUCTURED MESH USING PHI ON NODES
            2. COMPUTE GRAD(PHI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
            3. OBTAIN CANDIDATE POINTS ON FINER MESH FOR SOLUTIONS OF EQUATION     NORM(GRAD(PHI))^2 = 0
            4. USING A SOLVER, FIND SOLUTION OF  NORM(GRAD(PHI))^2 = 0   BY EVALUATING AN INTERPOLATION OF GRAD(PHI)
            5. CHECK HESSIAN AT SOLUTIONS
            6. INTERPOLATE VALUE OF PHI AT CRITICAL POINT
        """
        
        # 1. INTERPOLATE PHI VALUES ON A FINER STRUCTURED MESH USING PHI ON NODES
        # DEFINE FINER STRUCTURED MESH
        Mr = 60
        Mz = 80
        rfine = np.linspace(self.Xmin, self.Xmax, Mr)
        zfine = np.linspace(self.Ymin, self.Ymax, Mz)
        # INTERPOLATE PHI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine)
        PHIfine = griddata((self.X[:,0],self.X[:,1]), self.PHI.T[0], (Rfine, Zfine), method='cubic')
        
        # 2. COMPUTE NORM(GRAD(PHI)) WITH FINER MESH VALUES USING FINITE DIFFERENCES
        dr = (rfine[-1]-rfine[0])/Mr
        dz = (zfine[-1]-zfine[0])/Mz
        gradPHIfine = np.gradient(PHIfine,dr,dz)
        NORMgradPHIfine = np.zeros(np.shape(gradPHIfine)[1:])
        for i in range(Mr):
            for j in range(Mz):
                NORMgradPHIfine[j,i] = np.linalg.norm(np.array([gradPHIfine[0][j,i],gradPHIfine[1][j,i]])) 
        
        # 3. OBTAIN CANDIDATE POINTS ON FINER MESH FOR SOLUTIONS OF EQUATION     NORM(GRAD(PHI))^2 = 0
        X0 = np.array([self.R0,0])
        
        # 4. USING A GRADIENT DESCENT, FIND SOLUTION OF  NORM(GRAD(PHI))^2 = 0   BY EVALUATING AN INTERPOLATION OF GRAD(PHI)
        # INTERPOLATION OF GRAD(PHI)
        def gradPHI(X,Rfine,Zfine,gradPHIfine):
            dPHIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPHIfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPHIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPHIfine[1].flatten(), (X[0],X[1]), method='cubic')
            GRAD = np.array([dPHIdr,dPHIdz])
            return GRAD

        # GRADIENT DESCENT ROUTINE
        def gradient_descent(gradient, X0, alpha, itmax, tolerance, Rfine,Zfine,gradPHIfine):
            Xk = np.zeros([itmax,2])
            it = 0; TOL = 1
            Xk[it,:] = X0
            while TOL > tolerance and it < itmax:
                dX = -alpha * gradient(Xk[it,:], Rfine,Zfine,gradPHIfine)
                Xk[it+1,:] = Xk[it,:]+np.flip(dX)
                TOL = np.linalg.norm(Xk[it+1,:]-Xk[it,:])
                it += 1
            return it, TOL, Xk[:it,:] 
        
        # FIND MINIMUM USING GRADIENT DESCENT
        #alpha = 0.2
        #itmax = 50; tolerance = 1e-3
        #it, TOL, Xk = gradient_descent(gradPHI, X0, alpha, itmax, tolerance, Rfine,Zfine,gradPHIfine)
        #Xcrit = Xk[-1,:]
        
        sol = optimize.root(gradPHI, X0, args=(Rfine,Zfine,gradPHIfine))
        Xcrit = sol.x
        
        # 5. CHECK HESSIAN AT SOLUTIONS
        def EvaluateHESSIAN(X,gradPHIfine,Rfine,Zfine,dr,dz):
            # compute second derivatives on fine mesh
            dgradPHIdrfine = np.gradient(gradPHIfine[0],dr,dz)
            dgradPHIdzfine = np.gradient(gradPHIfine[1],dr,dz)
            # interpolate HESSIAN components on point 
            dPHIdrdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPHIdrfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPHIdzdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPHIdrfine[1].flatten(), (X[0],X[1]), method='cubic')
            dPHIdzdz = griddata((Rfine.flatten(),Zfine.flatten()), dgradPHIdzfine[1].flatten(), (X[0],X[1]), method='cubic')
            if dPHIdrdr*dPHIdzdz-dPHIdzdr**2 > 0:
                return "LOCAL EXTREMUM"
            else:
                return "SADDLE POINT"
            
        nature = EvaluateHESSIAN(Xcrit, gradPHIfine, Rfine, Zfine, dr, dz)
        
        # 6. INTERPOLATE VALUE OF PHI AT CRITICAL POINT
        def SearchElement(Elements,X,searchelements):
            """ Function which finds the element among the elements list containing the point with coordinates X. """
            for elem in searchelements:
                Xe = Elements[elem].Xe
                # Calculate the cross products (c1, c2, c3) for the point relative to each edge of the triangle
                c1 = (Xe[1,0]-Xe[0,0])*(X[1]-Xe[0,1])-(Xe[1,1]-Xe[0,1])*(X[0]-Xe[0,0])
                c2 = (Xe[2,0]-Xe[1,0])*(X[1]-Xe[1,1])-(Xe[2,1]-Xe[1,1])*(X[0]-Xe[1,0])
                c3 = (Xe[0,0]-Xe[2,0])*(X[1]-Xe[2,1])-(Xe[0,1]-Xe[2,1])*(X[0]-Xe[2,0])
                if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0): # INSIDE TRIANGLE
                    break
            return elem
        
        if nature == "LOCAL EXTREMUM":
            # FOR THE MAGNETIC AXIS VALUE PHI_0, THE LOCAL EXTREMUM SHOULD LIE INSIDE THE PLASMA REGION
            elem = SearchElement(self.Elements,Xcrit,self.PlasmaElems)
            self.PHI_0 = self.Elements[elem].ElementalInterpolation(Xcrit,self.PHI[self.Elements[elem].Te])
            self.PHI_X = 0
        else:
            elem = SearchElement(self.Elements,Xcrit,self.VacuumElems)
            self.PHI_X = self.Elements[elem].ElementalInterpolation(Xcrit,self.PHI[self.Elements[elem].Te])
            self.PHI_0 = 0
            
        return 
    
    
    def NormalisePHI(self):
        # NORMALISE SOLUTION OBTAINED FROM SOLVING CutFEM SYSTEM OF EQUATIONS USING CRITICAL PHI VALUES, PHI_0 AND PHI_X
        for i in range(self.Nn):
            self.PHI_NORM[i,1] = (self.PHI[i]-self.PHI_X)/np.abs(self.PHI_0-self.PHI_X)
        return 
    
    
    ##################################################################################################
    ################################ GLOBAL SYSTEM SOLVER ############################################
    ##################################################################################################
    
    def AssembleGlobalSystem(self):
        """ This routine assembles the global matrices derived from the discretised linear system of equations used the common Galerkin approximation. 
        Nonetheless, due to the unfitted nature of the method employed, integration in cut cells (elements containing the interface between plasma region 
        and vacuum region, defined by the level-set 0-contour) must be treated accurately. """
        
        # INITIALISE GLOBAL SYSTEM MATRICES
        self.LHS = np.zeros([self.Nn,self.Nn])
        self.RHS = np.zeros([self.Nn, 1])
        
        # ELEMENTS INSIDE AND OUTSIDE PLASMA REGION (ELEMENTS WHICH ARE NOT CUT)
        print("     Assemble non-cut elements...", end="")
        for elem in np.concatenate((self.PlasmaElems, self.VacuumElems, self.BoundaryElems), axis=0): 
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem] 
            
            ####### COMPUTE DOMAIN TERMS 
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.Ng2D])
            if ELEMENT.Dom < 0:
                # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                PHIg = ELEMENT.N @ ELEMENT.PHIe
                for ig in range(self.Elements[elem].Ng2D):
                    SourceTermg[ig] = self.mu0*ELEMENT.Xg2D[ig,0]*self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],PHIg[ig]) 
            # COMPUTE ELEMENTAL MATRICES
            ELEMENT.IntegrateElementalDomainTerms(SourceTermg,self.LHS,self.RHS)
            
        ####### COMPUTE BOUNDARY TERMS 
        for elem in self.BoundaryElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]
            # INTERPOLATE BOUNDARY PHI VALUE PHI_B ON BOUNDARY GAUSS INTEGRATION NODES
            PHI_Bg = np.zeros([ELEMENT.Nebound,ELEMENT.Ng1D])
            for edge in range(ELEMENT.Nebound):
                PHI_Bg[edge,:] = ELEMENT.Nbound[edge,:,:] @ ELEMENT.PHI_Be[edge,:]
            # COMPUTE ELEMENTAL MATRICES
            ELEMENT.IntegrateElementalBoundaryTerms(PHI_Bg,self.beta,self.LHS,self.RHS)
                
        print("Done!")
        
        print("     Assemble cut elements...", end="")
        # INTERFACE ELEMENTS (CUT ELEMENTS)
        for elem in self.InterElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]
            
            # NOW, EACH INTERFACE ELEMENT IS DIVIDED INTO SUBELEMENTS ACCORDING TO THE POSITION OF THE APPROXIMATED INTERFACE ->> TESSELLATION
            # ON EACH SUBELEMENT THE WEAK FORM IS INTEGRATED USING ADAPTED NUMERICAL INTEGRATION QUADRATURES
            ####### COMPUTE DOMAIN TERMS
            # LOOP OVER SUBELEMENTS 
            for SUBELEM in ELEMENT.SubElements:  
                # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
                SourceTermg = np.zeros([SUBELEM.Ng2D])
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PHIg = SUBELEM.N @ ELEMENT.PHIe
                    for ig in range(SUBELEM.Ng2D):
                        SourceTermg[ig] = self.mu0*SUBELEM.Xg2D[ig,0]*self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],PHIg[ig])
                
                # COMPUTE ELEMENTAL MATRICES
                SUBELEM.IntegrateElementalDomainTerms(SourceTermg,self.LHS,self.RHS)
                     
            ####### COMPUTE INTERFACE TERMS
            # COMPUTE INTERFACE CONDITIONS PHI_D
            ELEMENT.PHI_Dg = np.zeros([ELEMENT.Ng1D])
            for ig in range(ELEMENT.Ng1D):
                ELEMENT.PHI_Dg[ig] = self.SolutionCASE(ELEMENT.Xgint[ig,:])
                
            # COMPUTE ELEMENTAL MATRICES
            ELEMENT.IntegrateElementalInterfaceTerms(ELEMENT.PHI_Dg,self.beta,self.LHS,self.RHS)
            
        print("Done!") 
        return 
    
    def SolveSystem(self):
        # SOLVE LINEAR SYSTEM OF EQUATIONS AND OBTAIN PHI
        self.PHI = np.linalg.solve(self.LHS, self.RHS)
        return
    
    
    ##################################################################################################
    ############################### CONVERGENCE VALIDATION ###########################################
    ##################################################################################################
    
    def CheckConvergence(self,VALUES):
        
        if VALUES == "PHI_NORM":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_NORM[:,1]) > 0:
                L2residu = np.linalg.norm(self.PHI_NORM[:,1] - self.PHI_NORM[:,0])/np.linalg.norm(self.PHI_NORM[:,1])
            else: 
                L2residu = np.linalg.norm(self.PHI_NORM[:,1] - self.PHI_NORM[:,0])
            if L2residu < self.INT_TOL:
                self.converg_INT = True   # STOP INTERNAL WHILE LOOP 
            else:
                self.converg_INT = False
            
        elif VALUES == "PHI_B":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_B[:,1]) > 0:
                L2residu = np.linalg.norm(self.PHI_B[:,1] - self.PHI_B[:,0])/np.linalg.norm(self.PHI_B[:,1])
            else: 
                L2residu = np.linalg.norm(self.PHI_B[:,1] - self.PHI_B[:,0])
            if L2residu < self.EXT_TOL:
                self.converg_EXT = True   # STOP EXTERNAL WHILE LOOP 
            else:
                self.converg_EXT = False
        return 
    
    def UpdatePHI(self,VALUES):
        
        if VALUES == 'PHI_NORM':
            if self.converg_INT == False:
                self.PHI_NORM[:,0] = self.PHI_NORM[:,1]
            elif self.converg_INT == True:
                pass
        
        elif VALUES == 'PHI_B':
            if self.converg_EXT == False:
                self.PHI_B[:,0] = self.PHI_B[:,1]
                self.PHI_NORM[:,0] = self.PHI_NORM[:,1]
            elif self.converg_EXT == True:
                self.PHI_CONV = self.PHI_NORM[:,1]
        
        return
    
    def UpdateElementalPHI(self,VALUES):
        """ Function to update the values of PHI_NORM or PHI_B in elements """
        
        if VALUES == 'PHI_NORM':
            for element in self.Elements:
                element.PHIe = self.PHI_NORM[element.Te,0]  # TAKE VALUES OF ITERATION N
        
        elif VALUES == 'PHI_B':
            # FOR FIXED PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PHI_B = 0, THUS SO ARE ALL ELEMENTAL VALUES  ->> PHI_Be = 0
            if self.PLASMA_BOUNDARY == 'FIXED':  
                for elem in self.BoundaryElems:
                    self.Elements[elem].PHI_Be = np.zeros([self.Elements[elem].Nebound,self.n])
            
            # FOR FREE PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PHI_B VALUES ARE COMPUTED FROM THE GRAD-SHAFRANOV OPERATOR'S GREEN FUNCTION
            # IN ROUTINE COMPUTEPHI_B, HERE WE NEED TO SEND TO EACH BOUNDARY ELEMENT THE PHI_B VALUES OF THEIR CORRESPONDING BOUNDARY NODES
            elif self.PLASMA_BOUNDARY == 'FREE':  
                # FOR EACH NODE ON THE BOUNDARY
                for i, nodeglobal in enumerate(self.BoundaryNodes):
                    # 1. FIND THE BOUNDARY ELEMENTS IN WHICH THE BOUNDARY NODE IS LOCATED
                    elems_index = self.Tbound[np.where(self.Tbound[:,:-1] == nodeglobal)[0], -1]
                    for elem in elems_index:
                        # 2. FIND THE LOCAL NODE CORRESPONDING TO THE BUNDARY NODE
                        local_index = np.where(self.Elements[elem].Te == nodeglobal)[0][0]
                        # 3. ASSIGN VALUE TO ELEMENTAL ATTRIBUTE
                        self.Elements[elem].PHI_Be[local_index] = self.PHI_B[i,0]  # TAKE VALUES OF ITERATION N
        return
    
    
    ##################################################################################################
    ############################### OPERATIONS OVER GROUPS ###########################################
    ##################################################################################################
    
    ##################### INITIALISATION 
    
    def InitialGuess(self):
        """ This function computes the problem's initial guess. """
        PHI0 = np.zeros([self.Nn])
        for i in range(self.Nn):
            PHIexact = self.SolutionCASE(self.X[i,:])
            PHI0[i] = PHIexact*2*random()
        return PHI0
    
    def InitialLevelSet(self):
        """ COMPUTE THE INITIAL LEVEL-SET FUNCTION VALUES DESCRIBING THE INITIAL PLASMA REGION GEOMETRY. 
            -> FIXED-BOUNDARY PROBLEM: Use the analytical solution for the LINEAR case as initial Level-Set function. The plasma region is characterised by a negative value of Level-Set.
            -> FREE-BOUNDARY PROBLEM: DIFFERENT OPTIONS, WHERE THE PLASMA SHAPE IS DEFINED BY SOME CONTROL POINTS """
            
        self.LevelSet = np.zeros([self.Nn])
            
        if self.PLASMA_BOUNDARY == 'FIXED':
            # ADIMENSIONALISE MESH
            Xstar = self.X/self.R0
            coeffs = self.ComputeLinearSolutionCoefficients()
            for i in range(self.Nn):
                self.LevelSet[i] = Xstar[i,0]**4/8 + coeffs[0] + coeffs[1]*Xstar[i,0]**2 + coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
                
        elif self.PLASMA_BOUNDARY == 'FREE':
            self.LevelSet = self.F4E_LevelSet()
            
        return 
    
    def InitialiseElements(self):
        """ Function initialising attribute ELEMENTS which is a list of all elements in the mesh. """
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.LevelSet[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    def InitialisePHI(self):  
        """ INITIALISE PHI VECTORS WHERE THE DIFFERENT SOLUTIONS WILL BE STORED ITERATIVELY DURING THE SIMULATION AND COMPUTE INITIAL GUESS."""
        # INITIALISE PHI VECTORS
        self.PHI = np.zeros([self.Nn])            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PHI_NORM = np.zeros([self.Nn,2])     # NORMALISED PHI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PHI_B = np.zeros([self.Nnbound,2])   # VACUUM VESSEL BOUNDARY PHI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PHI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION FIELD
        # COMPUTE INITIAL GUESS AND STORE IT IN INTERNAL SOLUTION FOR N=0
        self.PHI_NORM[:,0] = self.InitialGuess()     
        return
    
    
    def Initialization(self):
        """ Routine which initialises all the necessary elements in the problem """
        # INITIALISE PHI VARIABLES
        print("     -> COMPUTE INITIAL GUESS...", end="")
        self.InitialisePHI()
        print('Done!')

        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialLevelSet()
        print('Done!')

        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...", end="")
        self.InitialiseElements()
        print('Done!')

        # CLASSIFY ELEMENTS  ->  OBTAIN PLASMAELEMS, VACUUMELEMS, INTERELEMS, BOUNDARYELEMS
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.ClassifyElements()
        print("Done!")

        # COMPUTE COMPUTATIONAL DOMAIN'S BOUNDARY EDGES
        print("     -> FIND BOUNDARY EDGES...", end="")
        self.ComputeBoundaryEdges()
        print("Done!")

        # COMPUTE COMPUTATIONAL DOMAIN'S BOUNDARY NORMALS
        print('     -> COMPUTE BOUNDARY NORMALS...', end="")
        self.ComputeBoundaryNormals()
        print('Done!')

        # COMPUTE INTERFACE LINEAR APPROXIMATION
        print("     -> APPROXIMATE INTERFACE...", end="")
        self.ComputeInterfaceApproximation()
        print("Done!")

        # COMPUTE INTERFACE APPROXIMATION NORMALS
        print('     -> COMPUTE INTERFACE NORMALS...', end="")
        self.ComputeInterfaceNormals()
        print('Done!')

        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE INITIAL VACUUM VESSEL BOUNDARY VALUES PHI_B...', end="")
        self.ComputePHI_B('INITIALIZATION')
        print('Done!')
        return

    
    ##################### OPERATIONS ON COMPUTATIONAL DOMAIN'S BOUNDARY EDGES #########################
    
    def ComputeBoundaryEdges(self):
        """ Identify the edges lying on the computational domain's boundary, for each element on the boundary. """
        for elem in self.BoundaryElems:
            self.Elements[elem].FindBoundaryEdges(self.Tbound)
        return
    
    def ComputeBoundaryNormals(self):
        for elem in self.BoundaryElems:
            self.Elements[elem].BoundaryNormal(self.Xmax,self.Xmin,self.Ymax,self.Ymin)
        self.CheckBoundaryNormalVectors()
        return
    
    def CheckBoundaryNormalVectors(self):
        for elem in self.BoundaryElems:
            for edge in range(self.Elements[elem].Nebound):
                Xebound = self.Elements[elem].Xe[self.Elements[elem].Tebound[edge,:],:]
                dir = np.array([Xebound[1,0]-Xebound[0,0], Xebound[1,1]-Xebound[0,1]]) 
                scalarprod = np.dot(dir,self.Elements[elem].NormalVec[edge,:])
            if scalarprod > 1e-10: 
                raise Exception('Dot product equals',scalarprod, 'for mesh element', elem, ": Normal vector not perpendicular")
        return
    
    ##################### OPERATIONS ON INTERFACE ON CUT ELEMENTS ################################
    
    def ComputeInterfaceApproximation(self):
        """ Compute the coordinates for the points describing the interface linear approximation. """
        for inter, elem in enumerate(self.InterElems):
            self.Elements[elem].InterfaceLinearApproximation()
            self.Elements[elem].interface = inter
        return
    
    def ComputeInterfaceNormals(self):
        for elem in self.InterElems:
            self.Elements[elem].InterfaceNormal()
        self.CheckInterfaceNormalVectors()
        return
    
    def CheckInterfaceNormalVectors(self):
        for elem in self.InterElems:
            dir = np.array([self.Elements[elem].Xeint[1,0]-self.Elements[elem].Xeint[0,0], self.Elements[elem].Xeint[1,1]-self.Elements[elem].Xeint[0,1]]) 
            scalarprod = np.dot(dir,self.Elements[elem].NormalVec)
            if scalarprod > 1e-10: 
                raise Exception('Dot product equals',scalarprod, 'for mesh element', elem, ": Normal vector not perpendicular")
        return
    
    ##################### COMPUTE NUMERICAL INTEGRATION QUADRATURES FOR EACH ELEMENT GROUP 
    
    def ComputeIntegrationQuadratures(self):
        """ ROUTINE WHERE THE NUMERICAL INTEGRATION QUADRATURES FOR ALL ELEMENTS IN THE MESH ARE PREPARED. """
        
        # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for elem in np.concatenate((self.PlasmaElems, self.VacuumElems, self.BoundaryElems), axis=0):
            self.Elements[elem].ComputeStandardQuadrature2D(self.QuadratureOrder)
            
        # FOR BOUNDARY ELEMENTS COMPUTE BOUNDARY QUADRATURE ENTITIES TO INTEGRATE OVER BOUNDARY EDGES
        for elem in self.BoundaryElems:
            self.Elements[elem].ComputeBoundaryQuadrature(self.QuadratureOrder)
            
        # COMPUTE MODIFIED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for elem in self.InterElems:
            self.Elements[elem].ComputeModifiedQuadratures(self.QuadratureOrder)
            
        # COMPUTE 1D NUMERICAL INTEGRATION QUADRATURES TO INTEGRATE ALONG SOLENOIDS
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        self.XIg1D, self.Wg1D, self.Ng1D = GaussQuadrature(0,self.QuadratureOrder)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.N1D, self.dNdxi1D, foo = EvaluateReferenceShapeFunctions(self.XIg1D, 0, self.QuadratureOrder-1, self.nsole)
        return
    
    
    ##################################################################################################
    ######################################## MAIN ALGORITHM ##########################################
    ##################################################################################################
    
    def EQUILI(self):
        # READ INPUT FILES
        print("READ INPUT FILES...")
        self.ReadMesh()
        self.ReadEQUILIdata()
        print('Done!')
        
        # INITIALIZATION
        print("INITIALIZATION...")
        self.Initialization()
        print('Done!')

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.converg_EXT = False
        self.it_EXT = 0
        self.it = 0
        while (self.converg_EXT == False and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = False
            self.it_INT = 0
            #****************************************
            self.UpdateElementalPHI('PHI_B')
            while (self.converg_INT == False and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                ##################################
                self.UpdateElementalPHI('PHI_NORM')   # UPDATE PHI_NORM VALUES IN CORRESPONDING ELEMENTS
                self.AssembleGlobalSystem()  
                self.SolveSystem()                    # 1. SOLVE CutFEM SYSTEM  ->> PHI
                self.ComputeCriticalPHI()             # 2. COMPUTE CRITICAL VALUES   PHI_0 AND PHI_X
                self.NormalisePHI()                   # 3. NORMALISE PHI RESPECT TO CRITICAL VALUES  ->> PHI_NORM 
                self.CheckConvergence('PHI_NORM')     # 4. CHECK CONVERGENCE OF PHI_NORM FIELD
                self.UpdatePHI('PHI_NORM')            # 5. UPDATE PHI_NORM VALUES 
                ##################################
            self.ComputePHI_B('ITERATION')            # COMPUTE BOUNDARY VALUES PHI_B WITH INTERNALLY CONVERGED PHI_NORM
            self.CheckConvergence('PHI_B')            # CHECK CONVERGENCE OF BOUNDARY PHI VALUES  (PHI_B)
            self.UpdatePHI('PHI_B')                   # UPDATE PHI_NORM AND PHI_B VALUES
            #****************************************
        print('SOLUTION CONVERGED')
        self.PlotSolution(self.PHI_CONV,colorbar=True)
        return
    
    
    ##################################################################################################
    ############################### RENDERING AND REPRESENTATION #####################################
    ##################################################################################################
    
    def PlotSolution(self,phi,colorbar=False):
        if len(np.shape(phi)) == 2:
            phi = phi[:,0]
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.tricontourf(self.X[:,0],self.X[:,1], phi, levels=30)
        if colorbar == False:
            plt.tricontour(self.X[:,0],self.X[:,1], phi, levels=[0], colors='k')
        else:
            plt.colorbar()
        plt.show()
        return
    
    
    def PlotMesh(self):
        Tmesh = self.T + 1
        # Plot nodes
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.plot(self.X[:,0],self.X[:,1],'.')
        for e in range(self.Ne):
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='black', linewidth=1)
        plt.show()
        return
    
    def PlotMeshClassifiedElements(self):
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.tricontourf(self.X[:,0],self.X[:,1], self.LevelSet, levels=30, cmap='plasma')

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
        # PLOT BOUNDARY ELEMENTS
        for e in self.BoundaryElems:
            for i in range(self.n):
                plt.plot([self.X[int(Tmesh[e,i])-1,0], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,0]], 
                        [self.X[int(Tmesh[e,i])-1,1], self.X[int(Tmesh[e,int((i+1)%self.n)])-1,1]], color='orange', linewidth=1)
        # PLOT BOUNDARY NODES
        for node in self.BoundaryNodes:
            plt.scatter(self.X[node,0], self.X[node,1], color='green',marker='o',s=40)
                    
        plt.tricontour(self.X[:,0],self.X[:,1], self.LevelSet, levels=[0], colors='green',linewidths=3)
        
        plt.show()
        return
    
    def PlotInterfaceNormalVectors(self):
        fig, axs = plt.subplots(1, 2, figsize=(14,10))
        
        axs[0].set_xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        axs[0].set_ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        axs[1].set_xlim(5.5,7)
        axs[1].set_ylim(2.5,3.5)

        for i in range(2):
            axs[i].tricontour(self.X[:,0],self.X[:,1], self.LevelSet, levels=[0], colors='green',linewidths=6)
            for elem in self.InterElems:
                if i == 0:
                    dl = 5
                    axs[i].plot(self.Elements[elem].Xeint[:,0],self.Elements[elem].Xeint[:,1], linestyle='-',color = 'red', linewidth = 2)
                else:
                    dl = 10
                    for j in range(self.Elements[elem].n):
                        plt.plot([self.Elements[elem].Xe[j,0], self.Elements[elem].Xe[int((j+1)%self.Elements[elem].n),0]], 
                                [self.Elements[elem].Xe[j,1], self.Elements[elem].Xe[int((j+1)%self.Elements[elem].n),1]], color='k', linewidth=1)
                    axs[i].plot(self.Elements[elem].Xeint[:,0],self.Elements[elem].Xeint[:,1], linestyle='-',marker='o',color = 'red', linewidth = 2)
                Xeintmean = np.array([np.mean(self.Elements[elem].Xeint[:,0]),np.mean(self.Elements[elem].Xeint[:,1])])
                axs[i].arrow(Xeintmean[0],Xeintmean[1],self.Elements[elem].NormalVec[0]/dl,self.Elements[elem].NormalVec[1]/dl,width=0.01)
                
        axs[1].set_aspect('equal')
        plt.show()
        return
    
    def PlotInterfaceValues(self):
        import matplotlib as mpl

        # COLLECT DATA
        Ng1D = self.Elements[0].Ng1D
        X = np.zeros([len(self.InterElems)*Ng1D,self.dim])
        PHID = np.zeros([len(self.InterElems)*Ng1D])
        for i, elem in enumerate(self.InterElems):
            X[Ng1D*i:Ng1D*(i+1),:] = self.Elements[elem].Xgint
            PHID[Ng1D*i:Ng1D*(i+1)] = self.Elements[elem].PHI_Dg

        fig, ax = plt.subplots(figsize=(7,10))
        ax.set_ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        ax.set_xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PHID.min(),PHID.max())
        linecolors = cmap(norm(PHID))
        ax.scatter(X[:,0],X[:,1],color = linecolors)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax)
        plt.show()
        
        return
    
    def PlotError(self,phi):
        if len(np.shape(phi)) == 2:
            phi = phi[:,0]
            
        error = np.zeros([self.Nn])
        for i in range(self.Nn):
            PHIexact = self.SolutionCASE(self.X[i,:])
            error[i] = np.abs(PHIexact-phi[i])
            
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.tricontourf(self.X[:,0],self.X[:,1], error, levels=30)
        #plt.tricontour(self.X[:,0],self.X[:,1], PHIexact, levels=[0], colors='k')
        plt.colorbar()

        plt.show()
        
        return