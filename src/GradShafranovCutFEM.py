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
    epsilon0 = 8.8542E-12        # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19                # J eV-1   Botlzmann constant

    def __init__(self,mesh_folder,EQU_case_file,ElementType,ElementOrder):
        # INPUT FILES:
        self.mesh_folder = mesh_folder
        self.MESH = mesh_folder[mesh_folder.rfind("/")+1:]
        self.case_file = EQU_case_file
        
        # DECLARE PROBLEM ATTRIBUTES
        self.PLASMA_BOUNDARY = None         # PLASMA BOUNDARY BEHAVIOUR: 'FIXED'  or  'FREE'
        self.PLASMA_GEOMETRY = None         # PLASMA REGION GEOMETRY: "FIRST_WALL" or "F4E" 
        self.PLASMA_CURRENT = None          # PLASMA CURRENT MODELISATION: "LINEAR", "NONLINEAR" or "PROFILES"
        self.VACUUM_VESSEL = None           # VACUUM VESSEL GEOMETRY: "COMPUTATIONAL_DOMAIN" or "FIRST_WALL"
        self.TOTAL_CURRENT = None           # TOTAL CURRENT IN PLASMA
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = None        # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.VacVessWallElems = None        # LIST OF CUT (OR NOT) ELEMENT'S INDEXES, CONTAINING VACUUM VESSEL FIRST WALL (OR COMPUTATIONAL DOMAIN'S BOUNDARY)
        self.ExteriorElems = None           # LIST OF CUT ELEMENT'S INDEXES LYING ON THE VACUUM VESSEL FIRST WALL EXTERIOR REGION
        self.NonCutElems = None             # LIST OF ALL NON CUT ELEMENTS
        self.CutElems = None                # LIST OF ALL CUT ELEMENTS
        self.Elements = None                # ARRAY CONTAINING ALL ELEMENTS IN MESH (PYTHON OBJECTS)
        self.PlasmaBoundLevSet = None       # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.VacVessWallLevSet = None       # VACUUM VESSEL FIRST WALL GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.PHI = None                     # PHI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PHI POINTS
        self.PHI_0 = None                   # PHI VALUE AT MAGNETIC AXIS MINIMA
        self.PHI_X = None                   # PHI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PHI_NORM = None                # NORMALISED PHI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PHI_B = None                   # VACUUM VESSEL WALL PHI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PHI_CONV = None                # CONVERGED NORMALISED PHI SOLUTION FIELD  
        
        # RESULTS FOR EACH ITERATION
        self.PHI_NORM_ALL = None
        self.PHI_crit_ALL = None 
        self.PlasmaBoundLevSet_ALL = None
        self.ElementalGroups_ALL = None
        
        # VACCUM VESSEL FIRST WALL GEOMETRY
        self.epsilon = None                 # PLASMA REGION ASPECT RATIO
        self.kappa = None                   # PLASMA REGION ELONGATION
        self.delta = None                   # PLASMA REGION TRIANGULARITY
        self.Rmax = None                    # PLASMA REGION MAJOR RADIUS
        self.Rmin = None                    # PLASMA REGION MINOR RADIUS
        self.R0 = None                      # PLASMA REGION MEAN RADIUS
        
        ###### FOR FREE-BOUNDARY PROBLEM
        # PARAMETERS FOR COILS
        self.Ncoils = None                  # TOTAL NUMBER OF COILS
        self.Xcoils = None                  # COILS' COORDINATE MATRIX 
        self.Icoils = None                  # COILS' CURRENT
        # PARAMETERS FOR SOLENOIDS
        self.Nsolenoids = None              # TOTAL NUMBER OF SOLENOIDS
        self.Xsolenoids = None              # SOLENOIDS' COORDINATE MATRIX
        self.Isolenoids = None              # SOLENOIDS' CURRENT
        # PRESSURE AND TOROIDAL FIELD PROFILES
        self.B0 = None                      # TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
        self.q0 = None                      # TOKAMAK SAFETY FACTOR
        self.P0 = None                      # PRESSURE PROFILE FACTOR
        self.n_p = None                     # EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION
        self.G0 = None                      # TOROIDAL FIELD FACTOR
        self.n_g = None                     # EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
        
        ########################
        # COMPUTATIONAL MESH
        self.ElType = ElementType           # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = ElementOrder         # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.numvertices = None             # NUMBER OF VERTICES PER ELEMENT (= 3 IF TRIANGULAR; = 4 IF QUADRILATERAL)
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
        
        
        self.NnFW = None                    # NUMBER OF NODES ON VACUUM VESSEL FIRST WALL GEOMETRY
        self.XFW = None                     # COORDINATES MATRIX FOR NODES ON VACCUM VESSEL FIRST WALL GEOMETRY
        
        # NUMERICAL TREATMENT PARAMETERS
        self.QuadratureOrder = None              # NUMERICAL INTEGRATION QUADRATURE ORDER
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
        self.gamma = None                        # PLASMA TOTAL CURRENT CORRECTION FACTOR
        #### BOUNDARY CONSTRAINTS
        self.beta = None                         # NITSCHE'S METHOD PENALTY TERM
        self.coeffs1W = []                       # TOKAMAK FIRST WALL LEVEL-0 CONTOUR COEFFICIENTS (LINEAR PLASMA MODEL CASE SOLUTION)
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
        if self.ElType == 1:
            self.numvertices = 3
        elif self.ElType == 2:
            self.numvertices = 4
        
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
                - PLASMA BOUNDARY (FIXED/FREE BOUNDARY)
                - PLASMA REGION GEOMETRY (USE 1srt WALL OR TRUE F4E SHAPE)
                - PLASMA CURRENT MODELISATION
                - VACUUM VESSEL CONSIDERED GEOMETRY
                - TOTAL CURRENT
                - GEOMETRICAL PARAMETERS
                - LOCATION AND CURRENT OF EXTERNAL COILS CONFINING THE PLASMA
                - PLASMA PROPERTIES
                - NUMERICAL TREATMENT PARAMETERS
                """
        
        #############################################
        # INTER-CODE FUNCTIONS TO READ INPUT PARAMETERS BY BLOCKS 
        
        def BlockProblemParameters(self,line):
            if line[0] == 'PLASMA_BOUNDARY:':          # READ PLASMA BOUNDARY CONDITION (FIXED OR FREE)
                self.PLASMA_BOUNDARY = line[1]
            elif line[0] == 'PLASMA_GEOMETRY:':        # READ PLASMA REGION GEOMETRY (VACUUM VESSEL FIRST WALL OR F4E TRUE SHAPE)
                self.PLASMA_GEOMETRY = line[1]
            elif line[0] == 'PLASMA_CURRENT:':         # READ MODEL FOR PLASMA CURRENT (LINEAR, NONLINEAR OR DEFINED USING PROFILES FOR PRESSURE AND TOROIDAL FIELD)
                self.PLASMA_CURRENT = line[1]
            elif line[0] == 'VACUUM_VESSEL:':          # READ VACUUM VESSEL GEOMETRY (RECTANGLE -> COMPUTATIONAL DOMAIN BOUNDARY ; FIRST_WALL -> USE FIRST WALL GEOMETRY)
                self.VACUUM_VESSEL = line[1]
            elif line[0] == 'TOTAL_CURRENT:':        # READ TOTAL PLASMA CURRENT
                self.TOTAL_CURRENT = float(line[1])
            return
        
        def BlockFirstWall(self,line):
            if line[0] == 'R_MAX:':          # READ TOKAMAK FIRST WALL MAJOR RADIUS 
                self.Rmax = float(line[1])
            elif line[0] == 'R_MIN:':        # READ TOKAMAK FIRST WALL MINOR RADIUS 
                self.Rmin = float(line[1])
            elif line[0] == 'EPSILON:':      # READ TOKAMAK FIRST WALL INVERSE ASPECT RATIO
                self.epsilon = float(line[1])
            elif line[0] == 'KAPPA:':        # READ TOKAMAK FIRST WALL ELONGATION 
                self.kappa = float(line[1])
            elif line[0] == 'DELTA:':        # READ TOKAMAK FIRST WALL TRIANGULARITY 
                self.delta = float(line[1])
            return
        
        def BlockF4E(self,line):
            # READ PLASMA SHAPE CONTROL POINTS
            if line[0] == 'R_SADDLE:':    # READ PLASMA REGION X_CENTER 
                self.R_SADDLE = float(line[1])
            elif line[0] == 'Z_SADDLE:':    # READ PLASMA REGION Y_CENTER 
                self.Z_SADDLE = float(line[1])
            elif line[0] == 'R_RIGHTMOST:':    # READ PLASMA REGION X_CENTER 
                self.R_RIGHTMOST = float(line[1])
            elif line[0] == 'Z_RIGHTMOST:':    # READ PLASMA REGION Y_CENTER 
                self.Z_RIGHTMOST = float(line[1])
            elif line[0] == 'R_LEFTMOST:':    # READ PLASMA REGION X_CENTER 
                self.R_LEFTMOST = float(line[1])
            elif line[0] == 'Z_LEFTMOST:':    # READ PLASMA REGION Y_CENTER 
                self.Z_LEFTMOST = float(line[1])
            elif line[0] == 'R_TOP:':    # READ PLASMA REGION X_CENTER 
                self.R_TOP = float(line[1])
            elif line[0] == 'Z_TOP:':    # READ PLASMA REGION Y_CENTER 
                self.Z_TOP = float(line[1])
            return
        
        def BlockExternalMagnets(self,line,i,j):
            if line[0] == 'N_COILS:':    # READ PLASMA REGION X_CENTER 
                self.Ncoils = int(line[1])
                self.Xcoils = np.zeros([self.Ncoils,self.dim])
                self.Icoils = np.zeros([self.Ncoils])
            elif line[0] == 'Rposi:' and i<self.Ncoils:    # READ i-th COIL X POSITION
                self.Xcoils[i,0] = float(line[1])
            elif line[0] == 'Zposi:' and i<self.Ncoils:    # READ i-th COIL Y POSITION
                self.Xcoils[i,1] = float(line[1])
            elif line[0] == 'Inten:' and i<self.Ncoils:    # READ i-th COIL INTENSITY
                self.Icoils[i] = float(line[1])
                i += 1
            # READ SOLENOID PARAMETERS:
            elif l[0] == 'N_SOLENOIDS:':    # READ PLASMA REGION X_CENTER 
                self.Nsolenoids = int(l[1])
                self.Xsolenoids = np.zeros([self.Nsolenoids,self.dim+1])
                self.Nturnssole = np.zeros([self.Nsolenoids])
                self.Isolenoids = np.zeros([self.Nsolenoids])
            elif line[0] == 'Rposi:' and j<self.Nsolenoids:    # READ j-th SOLENOID X POSITION
                self.Xsolenoids[j,0] = float(line[1])
            elif line[0] == 'Zlow:' and j<self.Nsolenoids:     # READ j-th SOLENOID Y POSITION
                self.Xsolenoids[j,1] = float(line[1])
            elif line[0] == 'Zup:' and j<self.Nsolenoids:      # READ j-th SOLENOID Y POSITION
                self.Xsolenoids[j,2] = float(line[1])
            elif line[0] == 'Inten:' and j<self.Nsolenoids:    # READ j-th SOLENOID INTENSITY
                self.Isolenoids[j] = float(line[1])
                j += 1
            return i, j
        
        def BlockProfiles(self,line):
            if line[0] == 'B0:':    # READ TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
                self.B0 = float(line[1])
            elif line[0] == 'q0:':    # READ TOKAMAK SAFETY FACTOR 
                self.q0 = float(line[1])
            elif line[0] == 'n_p:':    # READ EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION 
                self.n_p = float(line[1])
            elif line[0] == 'g0:':    # READ TOROIDAL FIELD PROFILE FACTOR
                self.G0 = float(line[1])
            elif line[0] == 'n_g:':    # READ EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
                self.n_g = float(line[1])
            return
        
        def BlockNumericalTreatement(self,line):
            if line[0] == 'QUADRATURE_ORDER:':   # READ NUMERICAL INTEGRATION QUADRATURE ORDER
                self.QuadratureOrder = int(line[1])
            elif line[0] == 'EXT_ITER:':         # READ MAXIMAL NUMBER OF ITERATION FOR EXTERNAL LOOP
                self.EXT_ITER = int(line[1])
            elif line[0] == 'EXT_TOL:':          # READ TOLERANCE FOR EXTERNAL LOOP
                self.EXT_TOL = float(line[1])
            elif line[0] == 'INT_ITER:':         # READ MAXIMAL NUMBER OF ITERATION FOR INTERNAL LOOP
                self.INT_ITER = int(line[1])
            elif line[0] == 'INT_TOL:':          # READ TOLERANCE FOR INTERNAL LOOP
                self.INT_TOL = float(line[1])
            elif line[0] == 'BETA:':             # READ NITSCHE'S METHOD PENALTY PARAMETER 
                self.beta = float(line[1])
            elif line[0] == 'RELAXATION:':       # READ AITKEN'S METHOD RELAXATION PARAMETER
                self.alpha = float(line[1])
            return
        
        ################################################
                
        print("     -> READ EQUILI DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        EQUILIDataFile = self.case_file +'.equ.dat'
        file = open(EQUILIDataFile, 'r') 
        i = 0; j = 0
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
                    
            if l:  # LINE NOT EMPTY
                # READ PROBLEM PARAMETERS
                BlockProblemParameters(self,l)
                # READ TOKAMAK FIRST WALL GEOMETRY PARAMETERS
                BlockFirstWall(self,l)
                # READ CONTROL POINTS COORDINATES FOR F4E PLASMA SHAPE
                if self.PLASMA_GEOMETRY == 'F4E':
                    BlockF4E(self,l)
                # READ PARAMETERS FOR PRESSURE AND TOROIDAL FIELD PROFILES
                if self.PLASMA_CURRENT == 'PROFILES':
                    BlockProfiles(self,l)
                # READ COIL PARAMETERS
                if self.PLASMA_BOUNDARY == 'FREE':
                    i,j = BlockExternalMagnets(self,l,i,j)
                # READ NUMERICAL TREATMENT PARAMETERS
                BlockNumericalTreatement(self,l)
                
        if self.PLASMA_BOUNDARY == self.VACUUM_VESSEL:
            raise Exception("PLASMA REGION GEOMETRY AND VACUUM VESSEL FIRST WALL GEOMETRY MUST BE DIFFERENT")
            
        # COMPUTE TOKAMAK MEAN RADIUS
        self.R0 = (self.Rmax+self.Rmin)/2
        # TOKAMAK'S 1rst WALL GEOMETRY COEFFICIENTS, USED ALSO FOR LINEAR PLASMA MODEL ANALYTICAL SOLUTION (INITIAL GUESS)
        self.coeffs1W = self.ComputeLinearSolutionCoefficients()
        
        if self.PLASMA_CURRENT == "PROFILES":
            # COMPUTE PRESSURE PROFILE FACTOR
            self.P0=self.B0*(self.kappa**2+1)/(self.mu0*self.R0**2*self.q0*self.kappa)
            
        # PREPARE MATRICES FOR STORING ALL RESULTS
        self.PHI_NORM_ALL = np.zeros([self.Nn, self.INT_ITER*self.EXT_ITER])
        self.PHI_crit_ALL = np.zeros([2, 3, self.INT_ITER*self.EXT_ITER])   # dim0 = 2 -> LOCAL EXTREMUM AND SADDLE POINT;  dim1 = 3 -> [PHI_crit_val, x_crit, y_crit]
        self.PlasmaBoundLevSet_ALL = np.zeros([self.Nn, self.INT_ITER*self.EXT_ITER])
        self.ElementalGroups_ALL = np.zeros([self.Ne, self.INT_ITER*self.EXT_ITER])
             
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
    
    def AnalyticalSolutionLINEAR(self,X):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE LINEAR PLASMA MODEL at point with coordinates X. """
        # DIMENSIONALESS COORDINATES
        Xstar = X/self.R0
        PHIexact = (Xstar[0]**4)/8 + self.coeffs1W[0] + self.coeffs1W[1]*Xstar[0]**2 + self.coeffs1W[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
        return PHIexact
            
    def AnalyticalSolutionNONLINEAR(self,X):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE MANUFACTURED NONLINEAR PLASMA MODEL at point with coordinates X. """
        # DIMENSIONALESS COORDINATES
        Xstar = X/self.R0 
        coeffs = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0] 
        PHIexact = np.sin(coeffs[0]*(Xstar[0]+coeffs[2]))*np.cos(coeffs[1]*Xstar[1])  
        return PHIexact
    
    
    ##################################################################################################
    ###################################### PLASMA CURRENT ############################################
    ##################################################################################################
    
    def Jphi(self,R,Z,PHI):
        
        if self.PLASMA_CURRENT == 'LINEAR':
            # NORMALIE COORDINATES
            Rstar = R/self.R0
            # COMPUTE  PLASMA CURRENT
            # self.coeffs = [D1 D2 D3]  for linear solution
            jphi = Rstar/self.mu0
                
        elif self.PLASMA_CURRENT == 'NONLINEAR': 
            # NORMALIE COORDINATES
            Rstar = R/self.R0
            Zstar = Z/self.R0
            # COMPUTE  PLASMA CURRENT
            Kr = 1.15*np.pi
            Kz = 1.15
            r0 = -0.5
            jphi = -((Kr**2+Kz**2)*PHI+(Kr/Rstar)*np.cos(Kr*(Rstar+r0))*np.cos(Kz*Zstar)+Rstar*((np.sin(Kr*
                    (Rstar+r0))*np.cos(Kz*Zstar))**2-PHI**2+np.exp(-np.sin(Kr*(Rstar+r0))*np.cos(Kz*Zstar))-np.exp(-PHI)))/(self.mu0*Rstar)
        
        elif self.PLASMA_CURRENT == "PROFILES":
            ## OPTION WITH GAMMA APPLIED TO funG AND WITHOUT denom
            jphi = -R * self.dPdphi(PHI) - 0.5*self.dG2dphi(PHI)/ (R*self.mu0)
            
        return jphi
    
    def Jphi_NORM(self,R,Z,PHI):
        """ Function which computes the plasma current source term on the right hand side of the Grad-Shafranov equation. """
        
        if self.PLASMA_CURRENT == 'LINEAR':
            # NORMALIE COORDINATES
            Rstar = R/self.R0
            # COMPUTE  PLASMA CURRENT
            # self.coeffs = [D1 D2 D3]  for linear solution
            jphi = Rstar/self.mu0
                
        elif self.PLASMA_CURRENT == 'NONLINEAR': 
            # NORMALIE COORDINATES
            Rstar = R/self.R0
            Zstar = Z/self.R0
            # COMPUTE  PLASMA CURRENT
            Kr = 1.15*np.pi
            Kz = 1.15
            r0 = -0.5
            jphi = -((Kr**2+Kz**2)*PHI+(Kr/Rstar)*np.cos(Kr*(Rstar+r0))*np.cos(Kz*Zstar)+Rstar*((np.sin(Kr*
                    (Rstar+r0))*np.cos(Kz*Zstar))**2-PHI**2+np.exp(-np.sin(Kr*(Rstar+r0))*np.cos(Kz*Zstar))-np.exp(-PHI)))/(self.mu0*Rstar)
        
        elif self.PLASMA_CURRENT == "PROFILES":
            ## OPTION WITH GAMMA APPLIED TO funG AND WITHOUT denom
            jphi = -R * self.dPdphi(PHI) - 0.5*self.dG2dphi_NORM(PHI)/ (R*self.mu0)
            
        return jphi
    
    ######## PLASMA PRESSURE MODELING
    
    def dPdphi(self,phi):
        # FUNCTION MODELING PLASMA PRESSURE DERIVATIVE PROFILE 
        """if self.it <= 1:
            denom = 1
        else:
            denom = self.PHI_X - self.PHI_0
        dp = -self.P0*self.n_p*(phi**(self.n_p-1))/denom"""
        dp = self.P0*self.n_p*(phi**(self.n_p-1))
        return dp
    
    ######## TOROIDAL FUNCTION MODELING
    
    def dG2dphi(self,phi):
        # FUNCTION MODELING TOROIDAL FIELD FUNCTION DERIVATIVE IN PLASMA REGION
        dg = (self.G0**2)*self.n_g*(phi**(self.n_g-1))
        return dg
    
    def dG2dphi_NORM(self,phi):
        # FUNCTION MODELING TOROIDAL FIELD FUNCTION DERIVATIVE IN PLASMA REGION
        """if self.it <= 1:
            denom = 1
        else:
            denom = self.PHI_X - self.PHI_0
        dg = -(self.G0**2)*self.gamma*self.n_g*(phi**(self.n_g-1))/denom"""
        dg = (self.G0**2)*0.05*self.n_g*(phi**(self.n_g-1))
        return dg
    
    ##################################################################################################
    ###################################### LEVEL-SET DESCRIPTION #####################################
    ##################################################################################################
    
    def F4E_PlasmaBoundLevSet(self):
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
        P0 = np.array([self.R_SADDLE, self.Z_SADDLE])
        P1 = np.array([self.R_RIGHTMOST, self.Z_RIGHTMOST])
        P2 = np.array([self.R_LEFTMOST, self.Z_LEFTMOST])
        P3 = np.array([self.R_TOP, self.Z_TOP])
        
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
    
    def ComputePHI_B(self):
        
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
            k= np.sqrt(4*Xb[0]*Xp[0]/((Xb[0]+Xp[0])**2 + (Xp[1]-Xb[1])**2))
            Greenfun = (1/(2*np.pi))*(np.sqrt(Xp[0]*Xb[0])/k)*((2-k**2)*ellipticK(k)-2*ellipticE(k))
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
                
        PHI_B = np.zeros([self.NnFW])    
        # FOR FIXED PLASMA BOUNDARY PROBLEM THE VACUUM VESSEL BOUNDARY VALUES PHI_B ARE IRRELEVANT ->> PHI_B = 0
    
        if self.PLASMA_BOUNDARY == 'FREE':
            k = 0
            # COMPUTE PHI_B VALUE ON EACH VACUUM VESSEL ELEMENT FIRST WALL INTERFACE INTEGRATION POINTS
            for element in self.VacVessWallElems:
                for edge in range(self.Elements[element].Neint):
                    for point in range(self.Elements[element].InterEdges[edge].Ngaussint):
                        # ISOLATE NODAL COORDINATES
                        Xnode = self.Elements[element].InterEdges[edge].Xgint[point,:]
        
                        # CONTRIBUTION FROM EXTERNAL COILS CURRENT 
                        for icoil in range(self.Ncoils): 
                            PHI_B[k] += self.mu0 * GreenFunction(Xnode,self.Xcoils[icoil,:]) * self.Icoils[icoil]
                
                        # CONTRIBUTION FROM EXTERNAL SOLENOIDS CURRENT  ->>  INTEGRATE OVER SOLENOID LENGTH 
                        for isole in range(self.Nsolenoids):
                            Xsole = np.array([[self.Xsolenoids[isole,0], self.Xsolenoids[isole,1]],[self.Xsolenoids[isole,0], self.Xsolenoids[isole,2]]])   # COORDINATES OF SOLENOID EDGES
                            Jsole = self.Isolenoids[isole]/np.linalg.norm(Xsole[0,:]-Xsole[1,:])   # SOLENOID CURRENT LINEAR DENSITY
                            # LOOP OVER GAUSS NODES
                            for ig in range(self.Ng1D):
                                # MAPP 1D REFERENCE INTEGRATION GAUSS NODES TO PHYSICAL SPACE ON SOLENOID
                                Xgsole = self.N1D[ig,:] @ Xsole
                                # COMPUTE DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID 
                                detJ1D = Jacobian1D(Xsole[0,:],Xsole[1,:],self.dNdxi1D[ig,:])
                                detJ1D = detJ1D*2*np.pi*np.mean(Xsole[:,0])
                                for l in range(self.nsole):
                                    PHI_B[k] += self.mu0 * GreenFunction(Xnode,Xgsole) * Jsole * self.N1D[ig,l] * detJ1D * self.Wg1D[ig]
                    
                        # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                        #   1. INTEGRATE IN PLASMA ELEMENTS
                        for elem in self.PlasmaElems:
                            # ISOLATE ELEMENT OBJECT
                            ELEMENT = self.Elements[elem]
                            # INTERPOLATE ELEMENTAL PHI ON PHYSICAL GAUSS NODES
                            PHIg = ELEMENT.N @ ELEMENT.PHIe
                            # LOOP OVER GAUSS NODES
                            for ig in range(ELEMENT.Ng2D):
                                for l in range(ELEMENT.n):
                                    PHI_B[k] += self.mu0 * GreenFunction(Xnode, ELEMENT.Xg2D[ig,:])*self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],
                                                                            PHIg[ig])*ELEMENT.N[ig,l]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]*self.gamma
                            
                        #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                        for elem in self.PlasmaBoundElems:
                            # ISOLATE ELEMENT OBJECT
                            ELEMENT = self.Elements[elem]
                            # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                            for SUBELEM in ELEMENT.SubElements:
                                if SUBELEM.Dom < 0:  # IN PLASMA REGION
                                    # INTERPOLATE ELEMENTAL PHI ON PHYSICAL GAUSS NODES
                                    PHIg = SUBELEM.N @ ELEMENT.PHIe
                                    # LOOP OVER GAUSS NODES
                                    for ig in range(SUBELEM.Ng2D):
                                        for l in range(SUBELEM.n):
                                            PHI_B[k] += self.mu0 * GreenFunction(Xnode, SUBELEM.Xg2D[ig,:])*self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],
                                                                    PHIg[ig])*SUBELEM.N[ig,l]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]*self.gamma               
                        k += 1
        return PHI_B
    
    ##################################################################################################
    ###################################### ELEMENTS DEFINITION #######################################
    ##################################################################################################
    
    def ClassifyElements(self):
        """ Function that separates the elements into 5 groups: 
                - PlasmaElems: elements inside the plasma region P(phi) where the plasma current is different from 0
                - VacuumElems: elements outside the plasma region P(phi) where the plasma current is 0
                - PlasmaBoundElems: ELEMENTS CONTAINING THE INTERFACE BETWEEN PLASMA AND VACUUM
                - VacVessWallElems: ELEMENTS CONTAINING THE VACUUM VESSEL FIRST WALL 
                - ExteriorElems: ELEMENTS WHICH ARE OUTSIDE OF THE VACUUM VESSEL FIRST WALL (NON-EXISTING IF FIRST WALL IS COMPUTATIONAL DOMAIN)
                """
                
        """ FOR HIGH ORDER ELEMENTS (QUADRATIC, CUBIC...), ELEMENTS LYING ON GEOMETRY BOUNDARIES OR INTERFACES MAY BE CLASSIFIED AS SUCH DUE TO
        THE LEVEL-SET SIGN ON NODES WHICH ARE NOT VERTICES OF THE ELEMENT ('HIGH ORDER' NODES). IN CASES WHERE ON SUCH NODES POSSES DIFFERENT SIGN, THIS MAY LEAD TO AN INTERFACE
        WHICH CUTS TWICE THE SAME ELEMENTAL EDGE, MEANING THE INTERFACE ENTERS AND LEAVES THROUGH THE SAME SEGMENT. THE PROBLEM WITH THAT IS THE SUBROUTINE 
        RESPONSIBLE FOR APPROXIMATING THE INTERFACE INSIDE ELEMENTS ONLY SEES THE LEVEL-SET VALUES ON THE VERTICES, BECAUSE IT DOES ONLY BOTHER ON WHETHER THE 
        ELEMENTAL EDGE IS CUT OR NOT. 
        IN LIGHT OF SUCH OCURRENCES, THE CLASSIFICATION OF ELEMENTS BASED ON LEVEL-SET SIGNS WILL BE IMPLEMENTED SUCH THAT ONLY THE VALUES ON THE VERTICES ARE
        TAKEN INTO ACCOUNT. THAT WAY, THIS CASES ARE ELUDED. ON THE OTHER HAND, WE NEED TO DETECT ALSO SUCH CASES IN ORDER TO MODIFY THE VALUES OF THE MESH 
        LEVEL-SET VALUES AND ALSO ON THE ELEMENTAL VALUES. """
        
        self.PlasmaElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS INSIDE PLASMA REGION
        self.VacuumElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = np.zeros([self.Ne], dtype=int)   # GLOBAL INDEXES OF ELEMENTS CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM 
        self.VacVessWallElems = np.zeros([self.Ne], dtype=int)   # GLOBAL INDEXES OF ELEMENTS CONTAINING THE VACUUM VESSEL FIRST WALL
        self.ExteriorElems = np.zeros([self.Ne], dtype=int)      # GLOBAL INDEXES OF ELEMENTS OUTSIDE THE VACUUM VESSEL FIRST WALL
        kplasm = 0
        kvacuu = 0
        kint = 0
        kbound = 0
        kext = 0
        
        def CheckElementalVerticesLevelSetSigns(LSe):
            region = None
            DiffHighOrderNodes = []
            # CHECK SIGN OF LEVEL SET ON ELEMENTAL VERTICES
            for i in range(self.numvertices-1):
                # FIND ELEMENTS LYING ON THE INTERFACE (LEVEL-SET VERTICES VALUES EQUAL TO 0 OR WITH DIFFERENT SIGN)
                if LSe[i] == 0:  # if node is on Level-Set 0 contour
                    region = 0
                    break
                elif np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between vertices values change -> INTERFACE ELEMENT
                    region = 0
                    break
                # FIND ELEMENTS LYING INSIDE A SPECIFIC REGION (LEVEL-SET VERTICES VALUES WITH SAME SIGN)
                else:
                    if i+2 == self.numvertices:   # if all vertices values have the same sign
                        # LOCATE ON WHICH REGION LIES THE ELEMENT
                        if np.sign(LSe[i+1]) > 0:   # all vertices values with positive sign -> EXTERIOR REGION ELEMENT
                            region = +1
                        else:   # all vertices values with negative sign -> INTERIOR REGION ELEMENT 
                            region = -1
                            
                        # CHECK LEVEL-SET SIGN ON ELEMENTAL 'HIGH ORDER' NODES
                        #for i in range(self.numvertices,self.n-self.numvertices):  # LOOP OVER NODES WHICH ARE NOT ON VERTICES
                        for i in range(self.numvertices,self.n):
                            if np.sign(LSe[i]) != np.sign(LSe[0]):
                                DiffHighOrderNodes.append(i)
                
            return region, DiffHighOrderNodes
            
        for ielem in range(self.Ne):
            regionplasma, DHONplasma = CheckElementalVerticesLevelSetSigns(self.Elements[ielem].PlasmaLSe)    # check elemental vertices PLASMA INTERFACE level-set signs
            regionvessel, DHONvessel = CheckElementalVerticesLevelSetSigns(self.Elements[ielem].VacVessLSe)   # check elemental vertices VACUUM VESSEL FIRST WALL level-set signs
            if regionplasma < 0:   # ALL PLASMA LEVEL-SET NODAL VALUES NEGATIVE -> PLASMA ELEMENT 
                self.PlasmaElems[kplasm] = ielem
                self.Elements[ielem].Dom = -1
                kplasm += 1
            elif regionplasma == 0:  # DIFFERENT SIGN IN PLASMA LEVEL-SET NODAL VALUES -> PLASMA/VACUUM INTERFACE ELEMENT
                self.PlasmaBoundElems[kint] = ielem
                self.Elements[ielem].Dom = 0
                kint += 1
            elif regionplasma > 0: # ALL PLASMA LEVEL-SET NODAL VALUES POSITIVE -> VACUUM ELEMENT
                if regionvessel < 0:  # ALL VACUUM VESSEL LEVEL-SET NODAL VALUES NEGATIVE -> REGION BETWEEN PLASMA/VACUUM INTERFACE AND FIRST WALL
                    self.VacuumElems[kvacuu] = ielem
                    self.Elements[ielem].Dom = +1
                    kvacuu += 1
                elif regionvessel == 0: # DIFFERENT SIGN IN VACUUM VESSEL LEVEL-SET NODAL VALUES -> FIRST WALL ELEMENT
                    self.VacVessWallElems[kbound] = ielem
                    self.Elements[ielem].Dom = +2
                    kbound += 1
                elif regionvessel > 0:  # ALL VACUUM VESSEL LEVEL-SET NODAL VALUES POSITIVE -> EXTERIOR ELEMENT
                    self.ExteriorElems[kext] = ielem
                    self.Elements[ielem].Dom = +3
                    kext += 1
                    
            # IF THERE EXISTS 'HIGH-ORDER' NODES WITH DIFFERENT PLASMA LEVEL-SET SIGN
            if DHONplasma:
                for inode in DHONplasma:  # LOOP OVER LOCAL INDICES 
                    self.Elements[ielem].PlasmaLSe[inode] *= -1 
                    self.PlasmaBoundLevSet[self.Elements[ielem].Te[inode]] *= -1
                     
            # IF THERE EXISTS 'HIGH-ORDER' NODES WITH DIFFERENT VACUUM VESSEL LEVEL-SET SIGN
            if DHONvessel:
                for inode in DHONvessel:  # LOOP OVER LOCAL INDICES 
                    self.Elements[ielem].VacVessLSe[inode] *= -1 
                    self.VacVessWallLevSet[self.Elements[ielem].Te[inode]] *= -1
        
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.PlasmaBoundElems = self.PlasmaBoundElems[:kint]
        self.VacVessWallElems = self.VacVessWallElems[:kbound]
        self.ExteriorElems = self.ExteriorElems[:kext]
        
        # GATHER NON-CUT ELEMENTS AND CUT ELEMENTS 
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            self.NonCutElems = np.concatenate((self.PlasmaElems, self.VacuumElems, self.VacVessWallElems), axis=0)
            self.CutElems = self.PlasmaBoundElems
        else:
            self.NonCutElems = np.concatenate((self.PlasmaElems, self.VacuumElems, self.ExteriorElems), axis=0)
            self.CutElems = np.concatenate((self.PlasmaBoundElems, self.VacVessWallElems), axis=0)
        return
    
    def ObtainClassification(self):
        """ Function which produces an array where the different values code for the groups in which elements are classified. """
        Classification = np.zeros([self.Ne])
        for elem in self.PlasmaElems:
            Classification[elem] = -1
        for elem in self.PlasmaBoundElems:
            Classification[elem] = 0
        for elem in self.VacuumElems:
            Classification[elem] = +1
        for elem in self.VacVessWallElems:
            Classification[elem] = +2
        for elem in self.ExteriorElems:
            Classification[elem] = +3
            
        return Classification
    
    
    ##################################################################################################
    ############################### SOLUTION NORMALISATION ###########################################
    ##################################################################################################
    
    # SEARCH ELEMENT CONTAINING POINT IN MESH
    def SearchElement(self,X,searchelements):
        # Function which finds the element among the elements list containing the point with coordinates X. 
        
        if self.ElType == 1: # FOR TRIANGULAR ELEMENTS
            for elem in searchelements:
                Xe = self.Elements[elem].Xe
                # Calculate the cross products (c1, c2, c3) for the point relative to each edge of the triangle
                c1 = (Xe[1,0]-Xe[0,0])*(X[1]-Xe[0,1])-(Xe[1,1]-Xe[0,1])*(X[0]-Xe[0,0])
                c2 = (Xe[2,0]-Xe[1,0])*(X[1]-Xe[1,1])-(Xe[2,1]-Xe[1,1])*(X[0]-Xe[1,0])
                c3 = (Xe[0,0]-Xe[2,0])*(X[1]-Xe[2,1])-(Xe[0,1]-Xe[2,1])*(X[0]-Xe[2,0])
                if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0): # INSIDE TRIANGLE
                    return elem
        elif self.ElType == 2: # FOR QUADRILATERAL ELEMENTS
            for elem in searchelements:
                Xe = self.Elements[elem].Xe
                # This algorithm counts how many times a ray starting from the point intersects the edges of the quadrilateral. 
                # If the count is odd, the point is inside; otherwise, it is outside.
                inside = False
                for i in range(4):
                    if ((Xe[i,1] > X[1]) != (Xe[(i+1)%4,1]>X[1])) and (X[0]<(Xe[(i+1)%4,0]-Xe[i,0])*(X[1]-Xe[i,1])/(Xe[(i+1)%4,1]-Xe[i,1])+Xe[i,0]):
                        inside = not inside
                if inside:
                    return elem
    
    
    def ComputeCriticalPHI(self,PHI):
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
        # INTERPOLATION OF GRAD(PHI)
        def gradPHI(X,Rfine,Zfine,gradPHIfine):
            dPHIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPHIfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPHIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPHIfine[1].flatten(), (X[0],X[1]), method='cubic')
            GRAD = np.array([dPHIdr,dPHIdz])
            return GRAD
        
        # EVALUATE HESSIAN MATRIX ENTRIES
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
            
        
        # 1. INTERPOLATE PHI VALUES ON A FINER STRUCTURED MESH USING PHI ON NODES
        # DEFINE FINER STRUCTURED MESH
        Mr = 60
        Mz = 80
        rfine = np.linspace(self.Xmin, self.Xmax, Mr)
        zfine = np.linspace(self.Ymin, self.Ymax, Mz)
        # INTERPOLATE PHI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine)
        PHIfine = griddata((self.X[:,0],self.X[:,1]), PHI.T[0], (Rfine, Zfine), method='cubic')
        
        # 2. DEFINE GRAD(PHI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
        dr = (rfine[-1]-rfine[0])/Mr
        dz = (zfine[-1]-zfine[0])/Mz
        gradPHIfine = np.gradient(PHIfine,dr,dz)

        """
        # 3. FIND SOLUTION OF  GRAD(PHI) = 0   NEAR MAGNETIC AXIS AND SADDLE POINT 
        Nr = 2
        Nz = 2
        # EXPLORATION ZONE 1 (LOOKING FOR MAGNETIC AXIS LOCAL EXTREMUM)
        rA0 = np.linspace(self.R0-0.5,self.R0+0.5,Nr)
        zA0 = np.linspace(-0.5,0.5,Nz)
        # EXPLORATION ZONE 2 (LOOKING FOR SADDLE POINT)
        rA1 = np.linspace(4,5,Nr)
        zA1 = np.linspace(-5,-4,Nz)

        Xcritvec = np.zeros([(Nr*Nz)*2,2])
        i = 0
        # EXPLORE ZONE 1
        for r0 in rA0:
            for z0 in zA0:
                X0 = np.array([r0,z0])
                sol = optimize.root(gradPHI, X0, args=(Rfine,Zfine,gradPHIfine))
                if sol.success == True:
                    Xcritvec[i,:] = sol.x
                    i += 1
        # EXPLOR ZONE 2
        for r0 in rA1:
            for z0 in zA1:
                X0 = np.array([r0,z0])
                sol = optimize.root(gradPHI, X0, args=(Rfine,Zfine,gradPHIfine))
                if sol.success == True:
                    Xcritvec[i,:] = sol.x
                    i += 1
                    
        Xcritvec = Xcritvec[:i,:]
        
        # DISCARD SIMILAR SOLUTIONS:
        # Round each value in the array to the fourth decimal place
        Xcritvec_rounded = np.round(Xcritvec, decimals=4)
        # Convert each row to a tuple and create a set to remove duplicates
        Xcritvec_final = {tuple(row) for row in Xcritvec_rounded}
        # Convert the set back to a NumPy array
        Xcritvec_final = np.array(list(Xcritvec_final))
        
        # 4. CHECK HESSIAN -> Caracterise critical point
        self.Xcrit = np.zeros([len(Xcritvec_final[:,0]),3])
        for i in range(len(Xcritvec_final[:,0])):
            self.Xcrit[i,:2] = Xcritvec_final[i,:]
            nature = EvaluateHESSIAN(Xcritvec_final[i,:], gradPHIfine, Rfine, Zfine, dr, dz)
            if nature == "LOCAL EXTREMUM":
                self.Xcrit[i,-1] = 1
            elif nature == "SADDLE POINT":
                self.Xcrit[i,-1] = -1
        
        # 5. INTERPOLATE VALUE OF PHI AT CRITICAL POINT
        for i in range(len(self.Xcrit[:,0])):
            if self.Xcrit[i,-1] == 1:  # LOCAL EXTREMUM ON MAGNETIC AXIS 
                # LOOK FOR ELEMENT CONTAINING LOCAL EXTREMUM
                elem = SearchElement(self.Elements,self.Xcrit[i,:2],self.PlasmaElems)
                # INTERPOLATE PHI VALUE ON CRITICAL POINT
                self.PHI_0 = self.Elements[elem].ElementalInterpolation(self.Xcrit[i,:],PHI[self.Elements[elem].Te]) 
            elif self.Xcrit[i,-1] == -1:   # SADDLE POINT
                # LOOK FOR ELEMENT CONTAINING LOCAL EXTREMUM
                elem = SearchElement(self.Elements,self.Xcrit[i,:2],np.concatenate((self.VacuumElems,self.VacVessWallElems),axis=0))
                # INTERPOLATE PHI VALUE ON CRITICAL POINT
                self.PHI_X = self.Elements[elem].ElementalInterpolation(self.Xcrit[i,:],PHI[self.Elements[elem].Te]) 
                
        if self.PLASMA_BOUNDARY == "FIXED":
            self.PHI_X = 0
        """
        
        # FIND SOLUTION OF  GRAD(PHI) = 0   NEAR MAGNETIC AXIS AND SADDLE POINT 
        if self.it == 1:
            self.Xcrit = np.zeros([2,2,3])  # [(iterations n, n+1), (extremum, saddle point), (R_crit,Z_crit,PHI_crit)]
            X0_extr = np.array([6,0])
            X0_saddle = np.array([5,-4])
        else:
            X0_extr = self.Xcrit[0,0,:-1]
            X0_saddle = self.Xcrit[0,1,:-1]
            
        # 3. LOOK FOR LOCAL EXTREMUM
        sol = optimize.root(gradPHI, X0_extr, args=(Rfine,Zfine,gradPHIfine))
        if sol.success == True:
            self.Xcrit[1,0,:-1] = sol.x
        else:
            raise Exception("LOCAL EXTREMUM NOT FOUND")
        # 4. CHECK HESSIAN LOCAL EXTREMUM
        # LOCAL EXTREMUM
        nature = EvaluateHESSIAN(self.Xcrit[1,0,:-1], gradPHIfine, Rfine, Zfine, dr, dz)
        if nature != "LOCAL EXTREMUM":
            print("ERROR IN LOCAL EXTREMUM HESSIAN")
        # 5. INTERPOLATE VALUE OF PHI AT LOCAL EXTREMUM
        # LOOK FOR ELEMENT CONTAINING LOCAL EXTREMUM
        elem = self.SearchElement(self.Xcrit[1,0,:-1],self.PlasmaElems)
        self.Xcrit[1,0,-1] = elem
        # INTERPOLATE PHI VALUE ON CRITICAL POINT
        self.PHI_0 = self.Elements[elem].ElementalInterpolation(self.Xcrit[1,0,:-1],PHI[self.Elements[elem].Te]) 
        print('LOCAL EXTREMUM AT ',self.Xcrit[1,0,:-1],' (ELEMENT ', elem,') WITH VALUE PHI_0 = ',self.PHI_0)
            
        if self.PLASMA_BOUNDARY == "FREE":
            # 3. LOOK FOR SADDLE POINT
            sol = optimize.root(gradPHI, X0_saddle, args=(Rfine,Zfine,gradPHIfine))
            if sol.success == True:
                self.Xcrit[1,1,:-1] = sol.x
            else:
                raise Exception("SADDLE POINT NOT FOUND")
            # 4. CHECK HESSIAN SADDLE POINT
            nature = EvaluateHESSIAN(self.Xcrit[1,1,:-1], gradPHIfine, Rfine, Zfine, dr, dz)
            if nature != "SADDLE POINT":
                print("ERROR IN SADDLE POINT HESSIAN")
            # 5. INTERPOLATE VALUE OF PHI AT SADDLE POINT
            # LOOK FOR ELEMENT CONTAINING SADDLE POINT
            elem = self.SearchElement(self.Xcrit[1,1,:-1],np.concatenate((self.VacuumElems,self.PlasmaBoundElems,self.PlasmaElems),axis=0))
            self.Xcrit[1,1,-1] = elem
            # INTERPOLATE PHI VALUE ON CRITICAL POINT
            self.PHI_X = self.Elements[elem].ElementalInterpolation(self.Xcrit[1,1,:-1],PHI[self.Elements[elem].Te]) 
            print('SADDLE POINT AT ',self.Xcrit[1,1,:-1],' (ELEMENT ', elem,') WITH VALUE PHI_X = ',self.PHI_X)
        else:
            self.Xcrit[1,1,:-1] = [self.Xmin,self.Ymin]
            self.PHI_X = 0
            
        return 
    
    
    def NormalisePHI(self):
        # NORMALISE SOLUTION OBTAINED FROM SOLVING CutFEM SYSTEM OF EQUATIONS USING CRITICAL PHI VALUES, PHI_0 AND PHI_X
        for i in range(self.Nn):
            self.PHI_NORM[i,1] = (self.PHI[i]-self.PHI_X)/np.abs(self.PHI_0-self.PHI_X)
            
        return 
    
    
    def ComputeTotalPlasmaCurrent(self):
        """ Function that computes de total toroidal current carried by the plasma """
        
        Tcurrent = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PHIg = ELEMENT.N @ ELEMENT.PHIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.Ng2D):
                # LOOP OVER ELEMENTAL NODES
                for i in range(ELEMENT.n):
                    Tcurrent += self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],PHIg[ig])*ELEMENT.N[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PHIg = SUBELEM.N @ ELEMENT.PHIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.Ng2D):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            Tcurrent += self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],PHIg[ig])*SUBELEM.N[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]
                            
        return Tcurrent
    
    def ComputeTotalPlasmaCurrentNormalization(self):
        """ Function that computes the correction factor so that the total current flowing through the plasma region is constant and equal to input file parameter TOTAL_CURRENT. """
        
        Tcurrent = self.ComputeTotalPlasmaCurrent()
        # COMPUTE TOTAL PLASMA CURRENT CORRECTION FACTOR            
        self.gamma = self.TOTAL_CURRENT/Tcurrent
        
        """Int1 = 0
        Int2 = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PHIg = ELEMENT.N @ ELEMENT.PHIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.Ng2D):
                # LOOP OVER ELEMENTAL NODES
                for i in range(ELEMENT.n):
                    Int1 += ELEMENT.Xg2D[ig,0]*(PHIg[ig]**(self.n_p-1))*ELEMENT.N[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]
                    Int2 += ((PHIg[ig]**(self.n_g-1))/ELEMENT.Xg2D[ig,0])*ELEMENT.N[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PHI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PHIg = SUBELEM.N @ ELEMENT.PHIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.Ng2D):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            Int1 += SUBELEM.Xg2D[ig,0]*(PHIg[ig]**(self.n_p-1))*SUBELEM.N[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]
                            Int2 += ((PHIg[ig]**(self.n_g-1))/SUBELEM.Xg2D[ig,0])*SUBELEM.N[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]
        
        if self.it <= 1:
            denom = 1
        else:
            denom = self.PHI_X - self.PHI_0
            
        #self.gamma = -2*self.mu0*(self.TOTAL_CURRENT+self.P0*self.n_p*Int1)/(self.n_g*Int2*self.G0**2)
        self.gamma = -2*self.mu0*(self.TOTAL_CURRENT*denom+self.P0*self.n_p*Int1)/(self.n_g*Int2*self.G0**2)"""
        
        return
    
    
    ##################################################################################################
    ###################### CONVERGENCE VALIDATION and VARIABLES UPDATING #############################
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
                
            print("Internal iteration = ",self.it_INT,", PHI_NORM residu = ", L2residu)
            print(" ")
            
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
                
            print("External iteration = ",self.it_EXT,", PHI_B residu = ", L2residu)
            print(" ")
        return 
    
    def UpdatePHI(self,VALUES):
        
        if VALUES == 'PHI_NORM':
            self.PHI_NORM[:,0] = self.PHI_NORM[:,1]
            self.Xcrit[0,:,:] = self.Xcrit[1,:,:]
            
            """if self.converg_INT == False:
                self.PHI_NORM[:,0] = self.PHI_NORM[:,1]
            elif self.converg_INT == True:
                pass"""
        
        elif VALUES == 'PHI_B':
            if self.converg_EXT == False:
                self.PHI_B[:,0] = self.PHI_B[:,1]
                self.PHI_NORM[:,0] = self.PHI_NORM[:,1]
            elif self.converg_EXT == True:
                self.PHI_CONV = self.PHI_NORM[:,1]
        
        return
    
    def UpdateElementalPHI(self):
        """ Function to update the values of PHI_NORM in all mesh elements """
        for element in self.Elements:
            element.PHIe = self.PHI_NORM[element.Te,0]  # TAKE VALUES OF ITERATION N
        return
    
    def UpdateElementalPHI_g(self,INTERFACE):
        """ FUNCTION WHICH UPDATES THE PHI VALUE CONSTRAINTS PHI_g ON ALL INTERFACE EDGES INTEGRATION POINTS. """
        
        if INTERFACE == 'PLASMA/VACUUM':
            for elem in self.PlasmaBoundElems:
                # FOR EACH PLASMA/VACUUM INTERFACE EDGE
                for edge in range(self.Elements[elem].Neint):
                    # ISOLATE ELEMENTAL EDGE
                    EDGE = self.Elements[elem].InterEdges[edge]
                    # COMPUTE INTERFACE CONDITIONS PHI_D
                    EDGE.PHI_g = np.zeros([EDGE.Ngaussint])
                    # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE EDGE
                    for point in range(EDGE.Ngaussint):
                        if self.PLASMA_CURRENT == 'LINEAR':
                            EDGE.PHI_g[point] = self.AnalyticalSolutionLINEAR(EDGE.Xgint[point,:])
                        elif self.PLASMA_CURRENT == 'NONLINEAR':
                            EDGE.PHI_g[point] = self.AnalyticalSolutionNONLINEAR(EDGE.Xgint[point,:])
                        elif self.PLASMA_CURRENT == 'PROFILES':
                            EDGE.PHI_g[point] = self.PHI_X
                            #ELEMENT.PHI_g[edge,point] = 0
                
        elif INTERFACE == "FIRST WALL":
            # FOR FIXED PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PHI_B = 0, THUS SO ARE ALL ELEMENTAL VALUES  ->> PHI_Be = 0
            if self.PLASMA_BOUNDARY == 'FIXED':  
                for elem in self.VacVessWallElems:
                    for edge in range(self.Elements[elem].Neint):
                        self.Elements[elem].InterEdges[edge].PHI_g = np.zeros([self.Elements[elem].InterEdges[edge].Ngaussint])
            
            # FOR FREE PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PHI_B VALUES ARE COMPUTED FROM THE GRAD-SHAFRANOV OPERATOR'S GREEN FUNCTION
            # IN ROUTINE COMPUTEPHI_B, HERE WE NEED TO SEND TO EACH BOUNDARY ELEMENT THE PHI_B VALUES OF THEIR CORRESPONDING BOUNDARY NODES
            elif self.PLASMA_BOUNDARY == 'FREE':  
                k = 0
                # FOR EACH ELEMENT CONTAINING THE VACUUM VESSEL FIRST WALL
                for elem in self.VacVessWallElems:
                    # FOR EACH FIRST WALL EDGE
                    for edge in range(self.Elements[elem].Neint):
                        # ISOLATE EDGE
                        EDGE = self.Elements[elem].InterEdges[edge]
                        # FOR EACH INTEGRATION POINT ON THE FIRST WALL EDGE
                        for point in range(EDGE.Ngaussint):
                            EDGE.PHI_g[point] = self.PHI_B[k,0]
                            k += 1
        return
    
    
    ##################################################################################################
    ############################### OPERATIONS OVER GROUPS ###########################################
    ##################################################################################################
    
    ##################### INITIALISATION 
    
    def InitialGuess(self):
        """ This function computes the problem's initial guess, which is taken as the LINEAR CASE SOLUTION WITH SOME RANDOM NOISE. """
        PHI0 = np.zeros([self.Nn])
        if self.PLASMA_CURRENT == 'LINEAR':      
            for i in range(self.Nn):
                PHI0[i] = self.AnalyticalSolutionLINEAR(self.X[i,:])*2*random()
        elif self.PLASMA_CURRENT == 'NONLINEAR':
            for i in range(self.Nn):
                PHI0[i] = self.AnalyticalSolutionNONLINEAR(self.X[i,:])*2*random()
        else:
            for i in range(self.Nn):
                PHI0[i] = self.AnalyticalSolutionLINEAR(self.X[i,:])*(-0.5)
        return PHI0
    
    def InitialiseLevelSets(self):
        """ COMPUTE THE INITIAL LEVEL-SET FUNCTION VALUES DESCRIBING:
            ->> THE INITIAL PLASMA REGION GEOMETRY: 
                    -> PLASMA REGION (INSIDE) CHARACTERISED BY NEGATIVE SIGN OF LEVEL-SET FUNCTION
                    -> PLASMA REGION CAN BE DEFINED EITHER BY F4E GEOMETRY OR WITH SAME SHAPE AS VACCUM VESSEL FIRST WALL
                    -> IF VACCUM VESSEL BOUNDARY IS TAKEN AS THE FIRST WALL, THE PLASMA REGION MUST BE DEFINED BY OTHER THAN THE FIRST WALL (INCOMPATIBILITY PROBLEM) 
            ->> THE FIXED VACUUM VESSEL FIRST WALL GEOMETRY:
                    -> ONLY NECESSARY WHEN VACUUM_VESSEL FIRST WALL IS OTHER THAN AS THE COMPUTATIONAL DOMAIN """
            
        self.PlasmaBoundLevSet = np.zeros([self.Nn])
        if self.PLASMA_GEOMETRY == 'FIRST_WALL':  # IN THIS CASE, THE PLASMA REGION SHAPE IS TAKEN AS THE SHAPE OF THE TOKAMAK'S FIRST WALL
            for i in range(self.Nn):
                self.PlasmaBoundLevSet[i] = self.AnalyticalSolutionLINEAR(self.X[i,:])
        elif self.PLASMA_GEOMETRY == "F4E":    # IN THIS CASE, THE PLASMA REGION SHAPE IS DESCRIBED USING THE F4E SHAPE CONTROL POINTS
            self.PlasmaBoundLevSet = self.F4E_PlasmaBoundLevSet()
            
        self.VacVessWallLevSet = (-1)*np.ones([self.Nn])
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            for node_index in self.BoundaryNodes:
                self.VacVessWallLevSet[node_index] = 0
        elif self.VACUUM_VESSEL == "FIRST_WALL":
            for i in range(self.Nn):
                self.VacVessWallLevSet[i] = self.AnalyticalSolutionLINEAR(self.X[i,:])
            
        return 
    
    def InitialiseElements(self):
        """ Function initialising attribute ELEMENTS which is a list of all elements in the mesh. """
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.PlasmaBoundLevSet[self.T[e,:]],
                                 self.VacVessWallLevSet[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    def InitialisePHI(self):  
        """ INITIALISE PHI VECTORS WHERE THE DIFFERENT SOLUTIONS WILL BE STORED ITERATIVELY DURING THE SIMULATION AND COMPUTE INITIAL GUESS."""
        ####### INITIALISE PHI_NORM UNKNOWN VECTORS
        # INITIALISE PHI VECTORS
        self.PHI = np.zeros([self.Nn])            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PHI_NORM = np.zeros([self.Nn,2])     # NORMALISED PHI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PHI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION FIELD
        
        # COMPUTE INITIAL GUESS AND STORE IT IN INTERNAL SOLUTION FOR N=0
        print('         -> COMPUTE INITIAL GUESS FOR PHI_NORM...', end="")
        self.PHI_NORM[:,0] = self.InitialGuess()  
        # ASSIGN INITIAL GUESS PHI VALUES TO EACH ELEMENT
        self.UpdateElementalPHI()
        print('Done!')   
        
        ####### INITIALISE ELEMENTAL CONSTRAINT VALUES ON PLASMA/VACUUM INTERFACE
        self.PHI_X = 0
        self.UpdateElementalPHI_g('PLASMA/VACUUM')
        
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        if self.PLASMA_CURRENT == "PROFILES":
            self.gamma = 1
            self.ComputeTotalPlasmaCurrentNormalization()
            print("Total plasma current normalization factor = ", self.gamma)
            Tcurrent = self.ComputeTotalPlasmaCurrent()
            print("Normalised total plasma current = ", Tcurrent*self.gamma)
        
        ####### INITIALISE PHI_B UNKNOWN VECTORS
        # NODES ON VACUUM VESSEL FIRST WALL GEOMETRY FOR WHICH TO EVALUATE PHI_B VALUES == GAUSS INTEGRATION NODES ON FIRST WALL EDGES
        self.NnFW = 0
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                self.NnFW += self.Elements[elem].InterEdges[edge].Ngaussint
        # INITIALISE PHI_B VECTOR
        self.PHI_B = np.zeros([self.NnFW,2])      # VACUUM VESSEL FIRST WALL PHI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        # FOR VACUUM VESSEL FIRST WALL ELEMENTS, INITIALISE ELEMENTAL ATTRIBUTE PHI_Bg (PHI VALUES ON VACUUM VESSEL FIRST WALL INTERFACE EDGES GAUSS INTEGRATION POINTS)
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                self.Elements[elem].InterEdges[edge].PHI_g = np.zeros([self.Elements[elem].InterEdges[edge].Ngaussint])   
            
        # COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PHI_B 
        print('         -> COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PHI_B...', end="")
        self.PHI_B[:,0] = self.ComputePHI_B()
        # ASSIGN INITIAL PHI_B VALUES TO VACUUM VESSEL ELEMENTS
        self.UpdateElementalPHI_g('FIRST WALL')
        print('Done!')    
        
        return
    
    
    def Initialization(self):
        """ Routine which initialises all the necessary elements in the problem """
        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialiseLevelSets()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...", end="")
        self.InitialiseElements()
        print('Done!')
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.ClassifyElements()
        print("Done!")

        # COMPUTE COMPUTATIONAL DOMAIN'S BOUNDARY EDGES
        print("     -> APPROXIMATE VACUUM VESSEL FIRST WALL...", end="")
        self.ComputeFirstWallApproximation()
        print("Done!")

        # COMPUTE PLASMA/VACUUM INTERFACE LINEAR APPROXIMATION
        print("     -> APPROXIMATE PLASMA/VACUUM INTERFACE...", end="")
        self.ComputePlasmaInterfaceApproximation()
        print("Done!")
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        # INITIALISE PHI UNKNOWNS
        print("     -> INITIALISE UNKNOWN VECTORS AND COMPUTE INITIAL GUESS...")
        self.InitialisePHI()
        print('     Done!')
        return  
    
    ##################### OPERATIONS ON COMPUTATIONAL DOMAIN'S BOUNDARY EDGES #########################
    
    def ComputeFirstWallApproximation(self):
        """ APPROXIMATE/IDENTIFY LINEAR EDGES CONFORMING THE VACUUM VESSEL FIRST WALL GEOMETRY ON EACH EDGE. COMPUTE NORMAL VECTORS FOR EACH EDGE. """
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            for elem in self.VacVessWallElems:
                # IDENTIFY COMPUTATIONAL DOMAIN'S BOUNDARIES CONFORMING VACUUM VESSEL FIRST WALL
                self.Elements[elem].ComputationalDomainBoundaryEdges(self.Tbound)  
                # COMPUTE OUTWARDS NORMAL VECTOR
                self.Elements[elem].ComputationalDomainBoundaryNormal(self.Xmax,self.Xmin,self.Ymax,self.Ymin)
        else:
            for inter, elem in enumerate(self.VacVessWallElems):
                # APPROXIMATE VACUUM VESSEL FIRST WALL GEOMETRY CUTTING ELEMENT 
                self.Elements[elem].InterfaceLinearApproximation(inter)  
                # COMPUTE OUTWARDS NORMAL VECTOR
                self.Elements[elem].InterfaceNormal()
        # CHECK NORMAL VECTORS ORTHOGONALITY RESPECT TO INTERFACE EDGES
        self.CheckInterfaceNormalVectors("FIRST WALL")  
        return
    
    def ComputePlasmaInterfaceApproximation(self):
        """ Compute the coordinates for the points describing the interface linear approximation. """
        for inter, elem in enumerate(self.PlasmaBoundElems):
            # APPROXIMATE PLASMA/VACUUM INTERACE GEOMETRY CUTTING ELEMENT 
            self.Elements[elem].InterfaceLinearApproximation(inter)
            # COMPUTE OUTWARDS NORMAL VECTOR
            self.Elements[elem].InterfaceNormal()
        self.CheckInterfaceNormalVectors("PLASMA BOUNDARY")
        return
    
    def CheckInterfaceNormalVectors(self,INTERFACE):
        if INTERFACE == "PLASMA BOUNDARY":
            elements = self.PlasmaBoundElems
        elif INTERFACE == "FIRST WALL":
            elements = self.VacVessWallElems
            
        for elem in elements:
            for edge in range(self.Elements[elem].Neint):
                # ISOLATE INTERFACE SEGMENT/EDGE
                EDGE = self.Elements[elem].InterEdges[edge]
                dir = np.array([EDGE.Xeint[1,0]-EDGE.Xeint[0,0], EDGE.Xeint[1,1]-EDGE.Xeint[0,1]]) 
                scalarprod = np.dot(dir,EDGE.NormalVec)
                if scalarprod > 1e-10: 
                    raise Exception('Dot product equals',scalarprod, 'for mesh element', elem, ": Normal vector not perpendicular")
        return
    
    ##################### COMPUTE NUMERICAL INTEGRATION QUADRATURES FOR EACH ELEMENT GROUP 
    
    def ComputeIntegrationQuadratures(self):
        """ ROUTINE WHERE THE INITIAL NUMERICAL INTEGRATION QUADRATURES FOR ALL ELEMENTS IN THE MESH ARE PREPARED. """
        
        # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for elem in self.NonCutElems:
            self.Elements[elem].ComputeStandardQuadrature2D(self.QuadratureOrder)
            
        # COMPUTE MODIFIED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for elem in self.CutElems:
            self.Elements[elem].ComputeModifiedQuadratures(self.QuadratureOrder)
        
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":   
            # FOR BOUNDARY ELEMENTS COMPUTE BOUNDARY QUADRATURE ENTITIES TO INTEGRATE OVER BOUNDARY EDGES
            for elem in self.VacVessWallElems:
                self.Elements[elem].ComputeComputationalDomainBoundaryQuadrature(self.QuadratureOrder)
                
        # COMPUTE 1D NUMERICAL INTEGRATION QUADRATURES TO INTEGRATE ALONG SOLENOIDS
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1D, self.Wg1D, self.Ng1D = GaussQuadrature(0,self.QuadratureOrder)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.N1D, self.dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1D, 0, self.QuadratureOrder-1, self.nsole)
        return
    
    #################### UPDATE EMBEDED METHOD ##############################
    
    def UpdateElements(self):
        """ FUNCTION WHERE THE DIFFERENT METHOD ENTITIES ARE RECOMPUTED ACCORDING TO THE EVOLUTION OF THE LEVEL-SET DEFINING THE PLASMA REGION. 
        THEORETICALY, THE ONLY ELEMENTS AFFECTED AND THUS NEED TO RECOMPUTE THEIR ENTITIES AS THE PLASMA REGION EVOLVES SHOULD BE:
                - PLASMA ELEMENTS
                - PLASMABOUNDARY ELEMENTS
                - VACUUM ELEMENTS. """
                
        if self.PLASMA_BOUNDARY == "FREE":
            # IN CASE WHERE THE NEW SADDLE POINT (N+1) CORRESPONDS (CLOSE TO) TO THE OLD SADDLE POINT, THEN THAT MEANS THAT THE PLASMA REGION
            # IS ALREADY WELL DEFINED BY THE OLD LEVEL-SET 
            
            if np.linalg.norm(self.Xcrit[1,1,:-1]-self.Xcrit[0,1,:-1]) < 0.5:
                return
            
            else:
                ###### UPDATE PLASMA REGION LEVEL-SET FUNCTION VALUES ACCORDING TO SOLUTION OBTAINED
                # RECALL THAT PLASMA REGION IS DEFINED BY NEGATIVE VALUES OF LEVEL-SET
                for i in range(self.Nn):  
                    if self.X[i,0] < 3 or self.X[i,1] < self.Xcrit[1,1,1]:
                        self.PlasmaBoundLevSet[i] = np.abs(self.PHI_NORM[i,1])
                    else:
                        self.PlasmaBoundLevSet[i] = -self.PHI_NORM[i,1]

                ###### UPDATE PLASMA REGION LEVEL-SET ELEMENTAL VALUES     
                for ELEMENT in self.Elements:
                    ELEMENT.PlasmaLSe = self.PlasmaBoundLevSet[self.T[ELEMENT.index,:]]
                    
                self.ClassifyElements()
                
                ###### RECOMPUTE PLASMA/VACUUM INTERFACE LINEAR APPROXIMATION and NORMAL VECTORS
                self.ComputePlasmaInterfaceApproximation()
                
                ###### RECOMPUTE NUMERICAL INTEGRATION QUADRATURES
                for elem in np.concatenate((self.PlasmaElems, self.VacuumElems), axis = 0):
                    self.Elements[elem].ComputeStandardQuadrature2D(self.QuadratureOrder)
                # COMPUTE MODIFIED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
                for elem in self.PlasmaBoundElems:
                    self.Elements[elem].ComputeModifiedQuadratures(self.QuadratureOrder)
                return
    
    
    ##################################################################################################
    ################################ CutFEM GLOBAL SYSTEM ############################################
    ##################################################################################################
    
    def AssembleGlobalSystem(self):
        """ This routine assembles the global matrices derived from the discretised linear system of equations used the common Galerkin approximation. 
        Nonetheless, due to the unfitted nature of the method employed, integration in cut cells (elements containing the interface between plasma region 
        and vacuum region, defined by the level-set 0-contour) must be treated accurately. """
        
        # INITIALISE GLOBAL SYSTEM MATRICES
        self.LHS = np.zeros([self.Nn,self.Nn])
        self.RHS = np.zeros([self.Nn, 1])
        
        # INTEGRATE OVER THE SURFACE OF ELEMENTS WHICH ARE NOT CUT BY ANY INTERFACE (STANDARD QUADRATURES)
        print("     Integrate over non-cut elements...", end="")
        for elem in self.NonCutElems: 
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]  
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.Ng2D])
            if ELEMENT.Dom < 0:
                # MAP PHI VALUES FROM ELEMENT NODES TO GAUSS NODES
                PHIg = ELEMENT.N @ ELEMENT.PHIe
                for ig in range(self.Elements[elem].Ng2D):
                    #SourceTermg[ig] = self.mu0*ELEMENT.Xg2D[ig,0]*self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],PHIg[ig])
                    SourceTermg[ig] = self.mu0*ELEMENT.Xg2D[ig,0]*self.Jphi(ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],PHIg[ig])*self.gamma 
            # COMPUTE ELEMENTAL MATRICES
            ELEMENT.IntegrateElementalDomainTerms(SourceTermg,self.LHS,self.RHS)
        print("Done!")
        
        # INTEGRATE OVER THE SURFACES OF SUBELEMENTS IN ELEMENTS CUT BY INTERFACES (MODIFIED QUADRATURES)
        print("     Integrate over cut-elements subelements...", end="")
        for elem in self.CutElems:
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
                        #SourceTermg[ig] = self.mu0*SUBELEM.Xg2D[ig,0]*self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],PHIg[ig])
                        SourceTermg[ig] = self.mu0*SUBELEM.Xg2D[ig,0]*self.Jphi(SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],PHIg[ig])*self.gamma
                # COMPUTE ELEMENTAL MATRICES
                SUBELEM.IntegrateElementalDomainTerms(SourceTermg,self.LHS,self.RHS)
        print("Done!")
        
        print("     Integrate along cut-elements interface edges...", end="")
        for elem in self.CutElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]
            # COMPUTE ELEMENTAL MATRICES
            ELEMENT.IntegrateElementalInterfaceTerms(self.beta,self.LHS,self.RHS)
        
        # IN THE CASE WHERE THE VACUUM VESSEL FIRST WALL CORRESPONDS TO THE COMPUTATIONAL DOMAIN'S BOUNDARY, ELEMENTS CONTAINING THE FIRST WALL ARE NOT CUT ELEMENTS 
        # BUT STILL WE NEED TO INTEGRATE ALONG THE COMPUTATIONAL DOMAIN'S BOUNDARY BOUNDARY 
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            for elem in self.VacVessWallElems:
                # ISOLATE ELEMENT 
                ELEMENT = self.Elements[elem]
                # COMPUTE ELEMENTAL MATRICES
                ELEMENT.IntegrateElementalInterfaceTerms(self.beta,self.LHS,self.RHS)
        print("Done!") 
        
        return
    
    def SolveSystem(self):
        # SOLVE LINEAR SYSTEM OF EQUATIONS AND OBTAIN PHI
        self.PHI = np.linalg.solve(self.LHS, self.RHS)
        return
    
    def StorePHIValues(self):
        self.PHI_NORM_ALL[:,self.it] = self.PHI_NORM[:,0]   
        if self.it > 0:
            self.PHI_crit_ALL[0,0,self.it] = self.PHI_0
            self.PHI_crit_ALL[0,1:,self.it] = self.Xcrit[1,0,:-1]
            self.PHI_crit_ALL[1,0,self.it] = self.PHI_X
            self.PHI_crit_ALL[1,1:,self.it] = self.Xcrit[1,1,:-1]
        return
    
    def StoreMeshConfiguration(self):
        self.PlasmaBoundLevSet_ALL[:,self.it] = self.PlasmaBoundLevSet
        self.ElementalGroups_ALL[:,self.it] = self.ObtainClassification()
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
        self.it = 0
        self.Initialization()
        print('Done!')

        self.PlotSolution(self.PHI_NORM[:,0])  # PLOT INITIAL SOLUTION

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.converg_EXT = False
        self.it_EXT = 0
        # STORE INITIAL VALUES
        self.StorePHIValues()
        self.StoreMeshConfiguration()
        while (self.converg_EXT == False and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = False
            self.it_INT = 0
            #****************************************
            while (self.converg_INT == False and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                self.StoreMeshConfiguration()
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                ##################################
                self.PlotClassifiedElements_2()
                # COMPUTE TOTAL PLASMA CURRENT CORRECTION FACTOR
                if self.PLASMA_CURRENT == "PROFILES":
                    #self.ComputeTotalPlasmaCurrentNormalization()
                    self.gamma = 1
                    print("Total plasma current normalization factor = ", self.gamma)
                    Tcurrent = self.ComputeTotalPlasmaCurrent()
                    print("Normalised total plasma current = ", Tcurrent*self.gamma)
                #self.PlotPROFILES()
                self.PlotJphi_JphiNORM()
                self.AssembleGlobalSystem()  
                self.SolveSystem()                          # 1. SOLVE CutFEM SYSTEM  ->> PHI
                self.ComputeCriticalPHI(self.PHI)           # 2. COMPUTE CRITICAL VALUES   PHI_0 AND PHI_X
                self.NormalisePHI()                         # 3. NORMALISE PHI RESPECT TO CRITICAL VALUES  ->> PHI_NORM 
                self.PlotPHI_PHINORM()
                self.CheckConvergence('PHI_NORM')           # 4. CHECK CONVERGENCE OF PHI_NORM FIELD
                self.UpdateElements()
                self.UpdatePHI('PHI_NORM')                  # 5. UPDATE PHI_NORM VALUES (PHI_NORM[:,0] = PHI_NORM[:,1])
                self.UpdateElementalPHI()                   # 6. UPDATE PHI_NORM VALUES IN CORRESPONDING ELEMENTS (ELEMENT.PHIe = PHI_NORM[ELEMENT.Te,0])
                self.UpdateElementalPHI_g("PLASMA/VACUUM")  # 7. UPDATE ELEMENTAL CONSTRAINT VALUES PHI_g FOR PLASMA/VACUUM INTERFACE
                self.StorePHIValues()
                
                ##################################
            self.ComputeTotalPlasmaCurrentNormalization()
            print("Total plasma current normalization factor = ", self.gamma)
            Tcurrent = self.ComputeTotalPlasmaCurrent()
            print("Normalised total plasma current = ", Tcurrent*self.gamma)
            print('COMPUTE VACUUM VESSEL FIRST WALL VALUES PHI_B...', end="")
            self.PHI_B[:,1] = self.ComputePHI_B()     # COMPUTE VACUUM VESSEL FIRST WALL VALUES PHI_B WITH INTERNALLY CONVERGED PHI_NORM
            print('Done!')
            self.CheckConvergence('PHI_B')            # CHECK CONVERGENCE OF VACUUM VESSEL FIEST WALL PHI VALUES  (PHI_B)
            self.UpdatePHI('PHI_B')                   # UPDATE PHI_NORM AND PHI_B VALUES
            self.UpdateElementalPHI_g("FIRST WALL")   # UPDATE ELEMENTAL CONSTRAINT VALUES PHI_g FOR VACUUM VESSEL FIRST WALL INTERFACE 
            #****************************************
        print('SOLUTION CONVERGED')
        #self.PlotSolution(self.PHI_CONV)
        return
    
    
    ##################################################################################################
    ############################### RENDERING AND REPRESENTATION #####################################
    ##################################################################################################
    
    def PlotSolution(self,phi):
        if len(np.shape(phi)) == 2:
            phi = phi[:,0]
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        axs.set_xlim(self.Xmin,self.Xmax)
        axs.set_ylim(self.Ymin,self.Ymax)
        a = axs.tricontourf(self.X[:,0],self.X[:,1], phi, levels=30)
        axs.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs.tricontour(self.X[:,0],self.X[:,1], phi, levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs)
        plt.show()
        return
    
    def PlotJphi_JphiNORM(self):
        
        fig, axs = plt.subplots(1, 2, figsize=(11,5))
        
        for i in range(2):
            axs[i].set_xlim(self.Xmin, self.Xmax)
            axs[i].set_ylim(self.Ymin, self.Ymax)
        
        Jphi = np.zeros([self.Nn])
        Jphi_norm = np.zeros([self.Nn])
        for i in range(self.Nn):
            Jphi[i] = self.Jphi(self.X[i,0],self.X[i,1],self.PHI_NORM[i,0])
            Jphi_norm[i] = Jphi[i]*self.gamma
        
        # LEFT PLOT: PLASMA CURRENT Jphi 
        a0 = axs[0].tricontourf(self.X[:,0],self.X[:,1], Jphi, levels=50)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].set_title("PLASMA CURRENT Jphi")
        plt.colorbar(a0, ax=axs[0])
        
        # RIGHT PLOT: NORMALIZED PLASMA CURRENT Jphi_NORM
        a1 = axs[1].tricontourf(self.X[:,0],self.X[:,1], Jphi_norm, levels=50)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[1].yaxis.set_visible(False)
        axs[1].set_title("NORMALIZED PLASMA CURRENT Jphi")
        plt.colorbar(a1, ax=axs[1])
        
        return
    
    
    def PlotPHI_PHINORM(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PHI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, AND PHI_NORM (NORMALISED PHI). """
        
        fig, axs = plt.subplots(1, 2, figsize=(11,5))
        for i in range(2):
            axs[i].set_xlim(self.Xmin, self.Xmax)
            axs[i].set_ylim(self.Ymin, self.Ymax)
        
        # CENTRAL PLOT: PHI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
        a0 = axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PHI[:,0], levels=50)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PHI[:,0], levels=[0], colors = 'black')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].set_title('POLOIDAL MAGNETIC FLUX PHI')
        plt.colorbar(a0, ax=axs[0])
        
        # RIGHT PLOT: PHI at iteration N+1 WITH NORMALISATION
        a1 = axs[1].tricontourf(self.X[:,0],self.X[:,1], self.PHI_NORM[:,1], levels=50)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PHI_NORM[:,1], levels=[0], colors = 'black')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[1].set_title('NORMALIZED POLOIDAL MAGNETIC FLUX PHI_NORM')
        axs[1].yaxis.set_visible(False)
        plt.colorbar(a1, ax=axs[1])
        
        ## PLOT LOCATION OF CRITICAL POINTS
        for i in range(2):
            # LOCAL EXTREMUM
            axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'x',color='red', s = 40, linewidths = 2)
            # SADDLE POINT
            axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'x',color='green', s = 40, linewidths = 2)
        
        ## PLOT ELEMENTS CONTAINING CRITICAL POINTS
        # LOCAL EXTREMUM
        ELEMENT = self.Elements[int(self.Xcrit[1,0,-1])]
        for j in range(ELEMENT.n):
            axs[0].plot([ELEMENT.Xe[j,0], ELEMENT.Xe[int((j+1)%ELEMENT.n),0]],[ELEMENT.Xe[j,1], ELEMENT.Xe[int((j+1)%ELEMENT.n),1]], color='red', linewidth=1) 
        # SADDLE POINT
        ELEMENT = self.Elements[int(self.Xcrit[1,1,-1])]
        for j in range(ELEMENT.n):
            axs[0].plot([ELEMENT.Xe[j,0], ELEMENT.Xe[int((j+1)%ELEMENT.n),0]],[ELEMENT.Xe[j,1], ELEMENT.Xe[int((j+1)%ELEMENT.n),1]], color='green', linewidth=1) 
        
        plt.show()
        return
    
    
    def InspectElement(self,element_index,INTERFACE,QUADRATURE):
        
        ELEM = self.Elements[element_index]
        Xmin = np.min(ELEM.Xe[:,0])-0.1
        Xmax = np.max(ELEM.Xe[:,0])+0.1
        Ymin = np.min(ELEM.Xe[:,1])-0.1
        Ymax = np.max(ELEM.Xe[:,1])+0.1
        if ELEM.ElType == 1:
            numedges = 3
        elif ELEM.ElType == 2:
            numedges = 4
            
        if ELEM.Dom == -1:
            color = 'red'
        elif ELEM.Dom == 0:
            color = 'gold'
        elif ELEM.Dom == 1:
            color = 'grey'
        elif ELEM.Dom == 2:
            color = 'cyan'
        elif ELEM.Dom == 3:
            color = 'black'
        
        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        axs[0].set_xlim(self.Xmin-0.25,self.Xmax+0.25)
        axs[0].set_ylim(self.Ymin-0.25,self.Ymax+0.25)
        axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PHI_NORM[:,1], levels=30, cmap='plasma')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PHI_NORM[:,1], levels=[0], colors = 'black')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        for iedge in range(numedges):
            axs[0].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=3)
        
        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=3)
        axs[1].scatter(ELEM.Xe[:,0],ELEM.Xe[:,1], marker='o', s=50)
        if INTERFACE:
            axs[1].scatter(ELEM.InterEdges[0].Xeint[:,0],ELEM.InterEdges[0].Xeint[:,1],marker='x',s=50)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg2D[:,0],ELEM.Xg2D[:,1],marker='x',c='black')
            else:
                for SUBELEM in ELEM.SubElements:
                    # PLOT SUBELEMENT EDGES
                    for i in range(numedges):
                        plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='black', linewidth=1)
                    # PLOT QUADRATURE INTEGRATION POINTS
                    plt.scatter(SUBELEM.Xg2D[:,0],SUBELEM.Xg2D[:,1],marker='x',c='grey')
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
        plt.ylim(self.Ymin-0.25,self.Ymax+0.25)
        plt.xlim(self.Xmin-0.25,self.Xmax+0.25)
        plt.tricontourf(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=30, cmap='plasma')

        # PLOT NODES
        plt.plot(self.X[:,0],self.X[:,1],'.',color='black')
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            for i in range(ELEMENT.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[int((i+1)%ELEMENT.n),0]],[ELEMENT.Xe[i,1], ELEMENT.Xe[int((i+1)%ELEMENT.n),1]], color='red', linewidth=1)
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            for i in range(ELEMENT.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[int((i+1)%ELEMENT.n),0]],[ELEMENT.Xe[i,1], ELEMENT.Xe[int((i+1)%ELEMENT.n),1]], color='gray', linewidth=1)
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            for i in range(ELEMENT.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[int((i+1)%ELEMENT.n),0]],[ELEMENT.Xe[i,1], ELEMENT.Xe[int((i+1)%ELEMENT.n),1]], color='black', linewidth=1)
        # PLOT PLASMA/VACUUM INTERFACE ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            for i in range(ELEMENT.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[int((i+1)%ELEMENT.n),0]],[ELEMENT.Xe[i,1], ELEMENT.Xe[int((i+1)%ELEMENT.n),1]], color='gold', linewidth=1)
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            for i in range(ELEMENT.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[int((i+1)%ELEMENT.n),0]],[ELEMENT.Xe[i,1], ELEMENT.Xe[int((i+1)%ELEMENT.n),1]], color='cyan', linewidth=1)
             
        # PLOT PLASMA/VACUUM INTERFACE 
        plt.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors='green',linewidths=3)
        # PLOT VACUUM VESSEL FIRST WALL
        plt.tricontour(self.X[:,0],self.X[:,1], self.VacVessWallLevSet, levels=[0], colors='orange',linewidths=3)
        
        plt.show()
        return
    
    def PlotClassifiedElements_2(self):
        plt.figure(figsize=(5,6))
        plt.ylim(self.Ymin-0.25,self.Ymax+0.25)
        plt.xlim(self.Xmin-0.25,self.Xmax+0.25)
        
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numvertices,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numvertices,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numvertices,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'black')
        # PLOT PLASMA/VACUUM INTERFACE ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numvertices,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gold')
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numvertices,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'cyan')
             
        # PLOT PLASMA/VACUUM INTERFACE 
        plt.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors='green',linewidths=3)
        # PLOT VACUUM VESSEL FIRST WALL
        plt.tricontour(self.X[:,0],self.X[:,1], self.VacVessWallLevSet, levels=[0], colors='orange',linewidths=3)
        
        plt.show()
        return
        
    
    def PlotNormalVectors(self):
        fig, axs = plt.subplots(1, 2, figsize=(14,10))
        
        axs[0].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        axs[0].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[1].set_xlim(5.5,7)
        if self.PLASMA_GEOMETRY == "FIRST_WALL":
            axs[1].set_ylim(2.5,3.5)
        elif self.PLASMA_GEOMETRY == "F4E":
            axs[1].set_ylim(1.8,3.5)

        for i in range(2):
            # PLOT PLASMA/VACUUM INTERFACE
            axs[i].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors='green',linewidths=6)
            # PLOT VACUUM VESSEL FIRST WALL
            axs[i].tricontour(self.X[:,0],self.X[:,1], self.VacVessWallLevSet, levels=[0], colors='orange',linewidths=6)
            # PLOT NORMAL LINEAR APPROXIMATION NORMAL VECTORS
            for elem in np.concatenate((self.PlasmaBoundElems,self.VacVessWallElems),axis=0):
                if i == 0:
                    dl = 5
                else:
                    dl = 10
                    for j in range(self.Elements[elem].n):
                        plt.plot([self.Elements[elem].Xe[j,0], self.Elements[elem].Xe[int((j+1)%self.Elements[elem].n),0]], 
                                [self.Elements[elem].Xe[j,1], self.Elements[elem].Xe[int((j+1)%self.Elements[elem].n),1]], color='k', linewidth=1)
                for edge in range(self.Elements[elem].Neint):
                    # ISOLATE EDGE
                    EDGE = self.Elements[elem].InterEdges[edge]
                    # PLOT INTERFACE APPROXIMATIONS
                    axs[0].plot(EDGE.Xeint[:,0],EDGE.Xeint[:,1], linestyle='-',color = 'red', linewidth = 2)
                    axs[1].plot(EDGE.Xeint[:,0],EDGE.Xeint[:,1], linestyle='-',marker='o',color = 'red', linewidth = 2)
                    # PLOT NORMAL VECTORS
                    Xeintmean = np.array([np.mean(EDGE.Xeint[:,0]),np.mean(EDGE.Xeint[:,1])])
                    axs[i].arrow(Xeintmean[0],Xeintmean[1],self.Elements[elem].NormalVec[edge,0]/dl,self.Elements[elem].NormalVec[edge,1]/dl,width=0.01)
                
        axs[1].set_aspect('equal')
        plt.show()
        return
    
    
    # PREPARE FUNCTION WHICH PLOTS AT THE SAME TIME PHI_Dg and PHI_Bg, that is plots PHI_g
    
    
    def PlotInterfaceValues(self):
        """ Function which plots the values PHI_g at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """
        import matplotlib as mpl

        # COLLECT PHI_g DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D,self.dim])
        PHI_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PHI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for elem in self.PlasmaBoundElems:
            for edge in range(self.Elements[elem].Neint):
                for point in range(self.Elements[elem].Ng1D):
                    X_Dg[k,:] = self.Elements[elem].Xgint[edge,point]
                    PHI_Dg[k] = self.Elements[elem].PHI_g[edge,point]
                    k += 1
            for node in range(self.Elements[elem].n):
                X_D[l,:] = self.Elements[elem].Xe[node,:]
                PHI_D[l] = self.PHI[self.Elements[elem].Te[node]]
                l += 1
            
        # COLLECT PHI_g DATA ON VACUUM VESSEL FIRST WALL  
        X_Bg = np.zeros([self.NnFW,self.dim])
        PHI_Bg = self.PHI_B[:,0]
        X_B = np.zeros([len(self.VacVessWallElems)*self.n,self.dim])
        PHI_B = np.zeros([len(self.VacVessWallElems)*self.n])
        k = 0
        l = 0
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                for point in range(self.Elements[elem].Ng1D):
                    X_Bg[k,:] = self.Elements[elem].Xgint[edge,point]
                    k += 1
            for node in range(self.Elements[elem].n):
                X_B[l,:] = self.Elements[elem].Xe[node,:]
                PHI_B[l] = self.PHI[self.Elements[elem].Te[node]]
                l += 1
            
        fig, axs = plt.subplots(1, 2, figsize=(14,7))
        # LEFT SUBPLOT: CONSTRAINT VALUES ON PHI
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[0].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PHI_Bg.min(),PHI_Bg.max())
        linecolors_Dg = cmap(norm(PHI_Dg))
        linecolors_Bg = cmap(norm(PHI_Bg))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)
        axs[0].scatter(X_Bg[:,0],X_Bg[:,1],color = linecolors_Bg)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[0])

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[1].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        norm = plt.Normalize(PHI_B.min(),PHI_B.max())
        linecolors_D = cmap(norm(PHI_D))
        linecolors_B = cmap(norm(PHI_B))
        axs[1].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        axs[1].scatter(X_B[:,0],X_B[:,1],color = linecolors_B)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])

        plt.show()
        return
    
    def PlotIntegrationQuadratures(self):
        
        plt.figure(figsize=(9,11))
        plt.ylim(self.Ymin-0.25,self.Ymax+0.25)
        plt.xlim(self.Xmin-0.25,self.Xmax+0.25)

        # PLOT NODES
        plt.plot(self.X[:,0],self.X[:,1],'.',color='black')
        Tmesh = self.T +1
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='red', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg2D[:,0],ELEMENT.Xg2D[:,1],marker='x',c='red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gray', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg2D[:,0],ELEMENT.Xg2D[:,1],marker='x',c='gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg2D[:,0],ELEMENT.Xg2D[:,1],marker='x',c='black')
            
        # PLOT PLASMA/VACUUM INTERFACE ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gold', linewidth=1)
            # PLOT SUBELEMENT EDGES AND INTEGRATION POINTS
            for SUBELEM in ELEMENT.SubElements:
                # PLOT SUBELEMENT EDGES
                for i in range(self.n):
                    plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='gold', linewidth=1)
                # PLOT QUADRATURE INTEGRATION POINTS
                plt.scatter(SUBELEM.Xg2D[:,0],SUBELEM.Xg2D[:,1],marker='x',c='gold')
            # PLOT INTERFACE LINEAR APPROXIMATION AND INTEGRATION POINTS
            for edge in range(ELEMENT.Neint):
                # PLOT INTERFACE LINEAR APPROXIMATION
                plt.plot(ELEMENT.InterEdges[edge].Xeint[:,0], ELEMENT.InterEdges[edge].Xeint[:,1], color='green', linewidth=1)
                # PLOT INTERFACE QUADRATURE
                plt.scatter(ELEMENT.InterEdges[edge].Xgint[:,0],ELEMENT.InterEdges[edge].Xgint[:,1],marker='o',c='green')
                
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='darkturquoise', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
                plt.scatter(ELEMENT.Xg2D[:,0],ELEMENT.Xg2D[:,1],marker='x',c='darkturquoise')
            else:
                for SUBELEM in ELEMENT.SubElements:
                    # PLOT SUBELEMENT EDGES
                    for i in range(self.n):
                        plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='darkturquoise', linewidth=1)
                    # PLOT QUADRATURE INTEGRATION POINTS
                    plt.scatter(SUBELEM.Xg2D[:,0],SUBELEM.Xg2D[:,1],marker='x',c='darkturquoise')
            # PLOT INTERFACE LINEAR APPROXIMATION AND INTEGRATION POINTS
            for edge in range(ELEMENT.Neint):
                # PLOT INTERFACE LINEAR APPROXIMATION
                plt.plot(ELEMENT.InterEdges[edge].Xeint[:,0], ELEMENT.InterEdges[edge].Xeint[:,1], color='orange', linewidth=1)
                # PLOT INTERFACE QUADRATURE
                plt.scatter(ELEMENT.InterEdges[edge].Xgint[:,0],ELEMENT.InterEdges[edge].Xgint[:,1],marker='o',c='orange')

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