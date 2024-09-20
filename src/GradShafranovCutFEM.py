""" This script contains the Python object defining a plasma equilibrium problem, modeled using the Grad-Shafranov equation
in an axisymmetrical system such as a tokamak. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import shutil
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
        self.MESH = mesh_folder[mesh_folder.rfind("TS-CUTFEM-")+10:]
        self.case_file = EQU_case_file
        self.CASE = EQU_case_file[EQU_case_file.rfind('/')+1:]
        
        # OUTPUT FILES
        self.outputdir = '../RESULTS/' + self.CASE + '-' + self.MESH
        self.PARAMS_file = None             # OUTPUT FILE CONTAINING THE SIMULATION PARAMETERS 
        self.PSI_file = None                # OUTPUT FILE CONTAINING THE PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.PSIcrit_file = None            # OUTPUT FILE CONTAINING THE CRITICAL PSI VALUES
        self.PSI_NORM_file = None           # OUTPUT FILE CONTAINING THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.PSI_B_file = None              # OUTPUT FILE CONTAINING THE PSI_B BOUNDARY VALUES
        self.RESIDU_file = None
        self.ElementsClassi_file = None     # OUTPUT FILE CONTAINING THE CLASSIFICATION OF MESH ELEMENTS
        self.PlasmaLevSetVals_file = None   # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.VacVessLevSetVals_file = None  # OUTPUT FILE CONTAINING THE VACUUM VESSEL BOUNDARY LEVEL-SET FIELD VALUES
        
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
        self.PSI = None                     # PSI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PSI POINTS
        self.PSI_0 = None                   # PSI VALUE AT MAGNETIC AXIS MINIMA
        self.PSI_X = None                   # PSI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PSI_NORM = None                # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_B = None                   # VACUUM VESSEL WALL PSI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_CONV = None                # CONVERGED NORMALISED PSI SOLUTION FIELD 
        self.residu_INT = None
        self.residu_EXT = None 
        
        # RESULTS FOR EACH ITERATION
        self.PSI_NORM_ALL = None
        self.PSI_crit_ALL = None 
        self.PlasmaBoundLevSet_ALL = None
        self.ElementalGroups_ALL = None
        
        # VACCUM VESSEL FIRST WALL GEOMETRY
        self.epsilon = None                 # PLASMA REGION ASPECT RATIO
        self.kappa = None                   # PLASMA REGION ELONGATION
        self.delta = None                   # PLASMA REGION TRIANGULARITY
        self.Rmax = None                    # PLASMA REGION MAJOR RADIUS
        self.Rmin = None                    # PLASMA REGION MINOR RADIUS
        self.R0 = None                      # PLASMA REGION MEAN RADIUS

        # PARAMETRISED INITIAL PLASMA EQUILIBRIUM GUESS
        self.CONTROL_POINTS = None          # NUMBER OF CONTROL POINTS
        self.R_SADDLE = None                # R COORDINATE OF ACTIVE SADDLE POINT
        self.Z_SADDLE = None                # Z COORDINATE OF ACTIVE SADDLE POINT
        self.R_RIGHTMOST = None             # R COORDINATE OF POINT ON THE RIGHT
        self.Z_RIGHTMOST = None             # Z COORDINATE OF POINT ON THE RIGHT
        self.R_LEFTMOST = None              # R COORDINATE OF POINT ON THE LEFT
        self.Z_LEFTMOST = None              # Z COORDINATE OF POINT ON THE LEFT
        self.R_TOP = None                  # R COORDINATE OF POINT ON TOP
        self.Z_TOP = None                  # Z COORDINATE OF POINT ON TOP
                
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
        
        self.activenodes = None
        self.coeffsZHENG = np.zeros([6])
        
        self.output_file = None
        self.ELMAT_file = None
        self.ELMAT_output = True
        self.GlobalSystem_output = False
        
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
        MeshDataFile = self.mesh_folder +'/'+ 'TS-CUTFEM-' + self.MESH +'.dom.dat'
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
        MeshFile = self.mesh_folder +'/'+ 'TS-CUTFEM-' + self.MESH +'.geo.dat'
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
            elif line[0] == 'PLASMA_CURRENT:':         # READ MODEL FOR PLASMA CURRENT (LINEAR, NONLINEAR, ZHENG OR DEFINED USING PROFILES FOR PRESSURE AND TOROIDAL FIELD)
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
            if line[0] == 'CONTROL_POINTS:':    # READ PLASMA REGION X_CENTER 
                self.CONTROL_POINTS = int(line[1])
            elif line[0] == 'R_SADDLE:':    # READ PLASMA REGION X_CENTER 
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
        #self.R0 = (self.Rmax+self.Rmin)/2
        self.R0 = self.Rmax
        # TOKAMAK'S 1rst WALL GEOMETRY COEFFICIENTS, USED ALSO FOR LINEAR PLASMA MODEL ANALYTICAL SOLUTION (INITIAL GUESS)
        self.coeffs1W = self.ComputeLinearSolutionCoefficients()
        
        if self.PLASMA_CURRENT == "PROFILES":
            # COMPUTE PRESSURE PROFILE FACTOR
            self.P0=self.B0*((self.kappa**2)+1)/(self.mu0*(self.R0**2)*self.q0*self.kappa)
            
        # PREPARE MATRICES FOR STORING ALL RESULTS
        self.PSI_NORM_ALL = np.zeros([self.Nn, self.INT_ITER*self.EXT_ITER+1])
        self.PSI_crit_ALL = np.zeros([2, 3, self.INT_ITER*self.EXT_ITER+1])   # dim0 = 2 -> LOCAL EXTREMUM AND SADDLE POINT;  dim1 = 3 -> [PSI_crit_val, x_crit, y_crit]
        self.PlasmaBoundLevSet_ALL = np.zeros([self.Nn, self.INT_ITER*self.EXT_ITER+1])
        self.ElementalGroups_ALL = np.zeros([self.Ne, self.INT_ITER*self.EXT_ITER+1])
        
        # LOOK FOR EXTREMUM IN LINEAR OR NONLINEAR ANALITYCAL SOLUTIONS IN ORDER TO NORMALISE THE CONSTRAINED VALUES ON THE INTERFACE FOR FIXED BOUNDARY PROBLEM
        self.PSIextr_analytical = 1.0
        """if self.PLASMA_BOUNDARY == 'FIXED' and self.PLASMA_CURRENT in ['LINEAR', 'NONLINEAR']:
            X0 = np.array([self.R0,0])
            match self.PLASMA_CURRENT:
                case "LINEAR":
                    sol = optimize.minimize(self.AnalyticalSolutionLINEAR, X0)
                    self.PSIextr_analytical = self.AnalyticalSolutionLINEAR(sol.x)
                case "NONLINEAR":
                    sol = optimize.minimize(self.AnalyticalSolutionMINUSNONLINEAR, X0)
                    self.PSIextr_analytical = self.AnalyticalSolutionNONLINEAR(sol.x)"""
        
        print('Done!')  
        return
    
    
    ##################################################################################################
    ############################# INITIAL GUESS AND SOLUTION CASE ####################################
    ##################################################################################################
    
    def ComputeLinearSolutionCoefficients(self):
        """ Computes the coeffients for the magnetic flux in the linear source term case, that is for 
                    GRAD-SHAFRANOV EQ:  DELTA*(PSI) = R^2   (plasma current is linear such that Jphi = R/mu0)
            for which the exact solution is 
                    PSI = R^4/8 + D1 + D2*R^2 + D3*(R^4-4*R^2*Z^2)
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
    
    def ComputeZhengSolutionCoefficients(self):
        """ Computes the coefficients for the Grad-Shafranov equation analytical solution proposed in ZHENG paper. """
        Ri = self.R0*(1-self.epsilon)  # PLASMA SHAPE EQUATORIAL INNERMOST POINT R COORDINATE
        Ro = self.R0*(1+self.epsilon)  # PLASMA SHAPE EQUATORIAL OUTERMOST POINT R COORDINATE
        a = (Ro-Ri)/2                  # PLASMA MINOR RADIUS
        Rt = self.R0 - self.delta*a    # PLASMA SHAPE HIGHEST POINT R COORDINATE
        Zt = self.kappa*a              # PLASMA SHAPE HIGHEST POINT Z COORDINATE
        
        # SET THE COEFFICIENT A2 TO 0 FOR SIMPLICITY
        self.coeffsZHENG[5] = 0
        # COMPUTE COEFFICIENT A1 BY IMPOSING A CONSTANT TOTAL TOROIDAL PLASMA CURRENT Ip
        #                   Jphi = (A1*R**2 - A2)/ R*mu0 
        # IF A2 = 0, WE HAVE THEN       Jphi = A1* (R/mu0)   THAT IS WHAT WE NEED TO INTEGRATE
        # HENCE,   A1 = Ip/integral(Jphi)
        def fun(X,PSI):
            return X[0]/self.mu0
        
        #self.coeffsZHENG[4] = self.TOTAL_CURRENT/self.PlasmaDomainIntegral(fun)
        
        self.coeffsZHENG[4] = -0.1
        
        # FOR COEFFICIENTS C1, C2, C3 AND C4, WE SOLVE A LINEAR SYSTEM OF EQUATIONS BASED ON THE PLASMA SHAPE GEOMETRY
        A = np.array([[1,Ri**2,Ri**4,np.log(Ri)*Ri**2],
                      [1,Ro**2,Ro**4,np.log(Ro)*Ro**2],
                      [1,Rt**2,(Rt**2-4*Zt**2)*Rt**2,np.log(Rt)*Rt**2-Zt**2],
                      [0,2,4*(Rt**2-2*Zt**2),2*np.log(Rt)+1]])
        
        b = np.array([[-(self.coeffsZHENG[4]*Ri**4)/8],
                      [-(self.coeffsZHENG[4]*Ro**4)/8],
                      [-(self.coeffsZHENG[4]*Rt**4)/8+(self.coeffsZHENG[5]*Zt**2)/2],
                      [-(self.coeffsZHENG[4]*Rt**2)/2]])
        
        coeffs = np.linalg.solve(A,b)
        self.coeffsZHENG[:4] = coeffs.T[0].tolist()
        return 
    
    def PSIAnalyticalSolution(self,X,MODEL):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE LINEAR PLASMA MODEL at point with coordinates X. """
        match MODEL:
            case "LINEAR":
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0
                # ANALYTICAL SOLUTION
                PSIexact = (Xstar[0]**4)/8 + self.coeffs1W[0] + self.coeffs1W[1]*Xstar[0]**2 + self.coeffs1W[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
                
            case "NONLINEAR":
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0 
                # ANALYTICAL SOLUTION
                coeffs = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0] 
                PSIexact = np.sin(coeffs[0]*(Xstar[0]+coeffs[2]))*np.cos(coeffs[1]*Xstar[1])  
                
            case "ZHENG":
                # ANALYTICAL SOLUTION
                PSIexact = self.coeffsZHENG[0]+self.coeffsZHENG[1]*X[0]**2+self.coeffsZHENG[2]*(X[0]**4-4*X[0]**2*X[1]**2)+self.coeffsZHENG[3]*(np.log(X[0])
                                    *X[0]**2-X[1]**2)+(self.coeffsZHENG[4]*X[0]**4)/8 - (self.coeffsZHENG[5]*X[1]**2)/2
        
        return PSIexact
    
    
    def AnalyticalSolutionLINEAR(self,X):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE LINEAR PLASMA MODEL at point with coordinates X. """
        # DIMENSIONALESS COORDINATES
        Xstar = X/self.R0
        # ANALYTICAL SOLUTION
        PSIexact = (Xstar[0]**4)/8 + self.coeffs1W[0] + self.coeffs1W[1]*Xstar[0]**2 + self.coeffs1W[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
        return PSIexact
            
    def AnalyticalSolutionNONLINEAR(self,X):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE MANUFACTURED NONLINEAR PLASMA MODEL at point with coordinates X. """
        # DIMENSIONALESS COORDINATES
        Xstar = X/self.R0 
        # ANALYTICAL SOLUTION
        coeffs = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0] 
        PSIexact = np.sin(coeffs[0]*(Xstar[0]+coeffs[2]))*np.cos(coeffs[1]*Xstar[1])  
        return PSIexact
    
    def AnalyticalSolutionZHENG(self,X):
        """ Function which computes the ANALYTICAL SOLUTION FOR THE MANUFACTURED PLASMA CURRENT MODEL PROPOSED IN PAPER ZHENG. """
        PSIexact = self.coeffsZHENG[0]+self.coeffsZHENG[1]*X[0]**2+self.coeffsZHENG[2]*(X[0]**4-4*X[0]**2*X[1]**2)+self.coeffsZHENG[3]*(X[0]**2*
                                    np.log(X[0])-X[1]**2)+(self.coeffsZHENG[4]*X[0]**4)/8 - (self.coeffsZHENG[5]*X[1]**2)/2
        return PSIexact
    
    
    
    ##################################################################################################
    ###################################### PLASMA CURRENT ############################################
    ##################################################################################################
    
    def Jphi(self,X,PSI):
        # COMPUTES THE SOURCE TERM Jphi, WHICH GOES IN THE GRAD-SHAFRANOV EQ. RIGHT-HAND-SIDE     mu0*R*Jphi
        match self.PLASMA_CURRENT:
            case 'LINEAR':
                # COMPUTE LINEAR MODEL PLASMA CURRENT
                Jphi = X[0]/self.mu0
            
            case 'NONLINEAR': 
                # COMPUTE NONLINEAR MODEL PLASMA CURRENT
                Kr = 1.15*np.pi
                Kz = 1.15
                r0 = -0.5
                Jphi = -((Kr**2+Kz**2)*PSI+(Kr/X[0])*np.cos(Kr*(X[0]+r0))*np.cos(Kz*X[1])+X[0]*((np.sin(Kr*(X[0]+r0))*np.cos(Kz*X[1]))**2
                                    -PSI**2+np.exp(-np.sin(Kr*(X[0]+r0))*np.cos(Kz*X[1]))-np.exp(-PSI)))/(self.mu0*X[0])
            
            case 'ZHENG':
                # COMPUTE PLASMA CURRENT MODEL BASED ON ZHENG PAPER
                Jphi = (self.coeffsZHENG[4]*X[0]**2 - self.coeffsZHENG[5])/ (X[0]*self.mu0)
        
            case "PROFILES":
                ## OPTION WITH GAMMA APPLIED TO funG AND WITHOUT denom
                Jphi = -X[0] * self.dPdPSI(PSI) - 0.5*self.dG2dPSI(PSI)/ (X[0]*self.mu0)
            
        return Jphi
    
    ######## PLASMA PRESSURE MODELING
    
    def dPdPSI(self,PSI):
        # FUNCTION MODELING PLASMA PRESSURE DERIVATIVE PROFILE 
        dp = self.P0*self.n_p*(PSI**(self.n_p-1))
        return dp
    
    ######## TOROIDAL FUNCTION MODELING
    
    def dG2dPSI(self,PSI):
        # FUNCTION MODELING TOROIDAL FIELD FUNCTION DERIVATIVE IN PLASMA REGION
        dg = (self.G0**2)*self.n_g*(PSI**(self.n_g-1))
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
    ################################# VACUUM VESSEL BOUNDARY PSI_B ###################################
    ##################################################################################################
    
    def ComputePSI_B(self):
        
        """ FUNCTION TO COMPUTE THE COMPUTATIONAL DOMAIN BOUNDARY VALUES FOR PSI, PSI_B, ON BOUNDARY ELEMENT. 
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
                
        PSI_B = np.zeros([self.NnFW])    
        # FOR FIXED PLASMA BOUNDARY PROBLEM THE VACUUM VESSEL BOUNDARY VALUES PSI_B ARE IRRELEVANT ->> PSI_B = 0
    
        if self.PLASMA_BOUNDARY == 'FREE':
            
            k = 0
            # COMPUTE PSI_B VALUE ON EACH VACUUM VESSEL ELEMENT FIRST WALL INTERFACE INTEGRATION POINTS
            for element in self.VacVessWallElems:
                for edge in range(self.Elements[element].Neint):
                    for point in range(self.Elements[element].InterEdges[edge].Ngaussint):
                        # ISOLATE NODAL COORDINATES
                        Xnode = self.Elements[element].InterEdges[edge].Xgint[point,:]
                        
                        # CONTRIBUTION FROM EXTERNAL COILS CURRENT 
                        for icoil in range(self.Ncoils): 
                            PSI_B[k] += self.mu0 * GreenFunction(Xnode,self.Xcoils[icoil,:]) * self.Icoils[icoil]
                        
                        # CONTRIBUTION FROM EXTERNAL SOLENOIDS CURRENT  ->>  INTEGRATE OVER SOLENOID LENGTH 
                        for isole in range(self.Nsolenoids):
                            Xsole = np.array([[self.Xsolenoids[isole,0], self.Xsolenoids[isole,1]],[self.Xsolenoids[isole,0], self.Xsolenoids[isole,2]]])   # COORDINATES OF SOLENOID EDGES
                            Jsole = self.Isolenoids[isole]/np.linalg.norm(Xsole[0,:]-Xsole[1,:])   # SOLENOID CURRENT LINEAR DENSITY
                            # LOOP OVER GAUSS NODES
                            for ig in range(self.Ng1D):
                                # MAPP 1D REFERENCE INTEGRATION GAUSS NODES TO PHYSICAL SPACE ON SOLENOID
                                Xgsole = self.N1D[ig,:] @ Xsole
                                # COMPUTE DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID 
                                detJ1D = Jacobian1D(Xsole[:,0],Xsole[:,1],self.dNdxi1D[ig,:])
                                #detJ1D = detJ1D*2*np.pi*np.mean(Xsole[:,0])
                                for l in range(self.nsole):
                                    PSI_B[k] += self.mu0 * GreenFunction(Xnode,Xgsole) * Jsole * self.N1D[ig,l] * detJ1D * self.Wg1D[ig]
                                    
                        # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                        #   1. INTEGRATE IN PLASMA ELEMENTS
                        for elem in self.PlasmaElems:
                            # ISOLATE ELEMENT OBJECT
                            ELEMENT = self.Elements[elem]
                            # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                            PSIg = ELEMENT.N @ ELEMENT.PSIe
                            # LOOP OVER GAUSS NODES
                            for ig in range(ELEMENT.Ng2D):
                                for l in range(ELEMENT.n):
                                    PSI_B[k] += self.mu0 * GreenFunction(Xnode, ELEMENT.Xg2D[ig,:])*self.Jphi(ELEMENT.Xg2D[ig,:],
                                                                PSIg[ig])*ELEMENT.N[ig,l]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]*self.gamma
                                              
                        #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                        for elem in self.PlasmaBoundElems:
                            # ISOLATE ELEMENT OBJECT
                            ELEMENT = self.Elements[elem]
                            # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                            for SUBELEM in ELEMENT.SubElements:
                                if SUBELEM.Dom < 0:  # IN PLASMA REGION
                                    # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                                    PSIg = SUBELEM.N @ ELEMENT.PSIe
                                    # LOOP OVER GAUSS NODES
                                    for ig in range(SUBELEM.Ng2D):
                                        for l in range(SUBELEM.n):
                                            PSI_B[k] += self.mu0 * GreenFunction(Xnode, SUBELEM.Xg2D[ig,:])*self.Jphi(SUBELEM.Xg2D[ig,:],
                                                                PSIg[ig])*SUBELEM.N[ig,l]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]*self.gamma   
                        k += 1
        return PSI_B
    
    ##################################################################################################
    ###################################### ELEMENTS DEFINITION #######################################
    ##################################################################################################
    
    def ClassifyElements(self):
        """ Function that separates the elements into 5 groups: 
                - PlasmaElems: elements inside the plasma region P(PSI) where the plasma current is different from 0
                - VacuumElems: elements outside the plasma region P(PSI) where the plasma current is 0
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
                print('yes plasma boundary')
                print(ielem)
                self.OLDplasmaBoundLevSet = self.PlasmaBoundLevSet.copy()
                for inode in DHONplasma:  # LOOP OVER LOCAL INDICES 
                    self.Elements[ielem].PlasmaLSe[inode] *= -1 
                    self.PlasmaBoundLevSet[self.Elements[ielem].Te[inode]] *= -1
                     
            # IF THERE EXISTS 'HIGH-ORDER' NODES WITH DIFFERENT VACUUM VESSEL LEVEL-SET SIGN
            if DHONvessel:
                print('yes vacuum vessel')
                print(ielem)
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
        Classification = np.zeros([self.Ne],dtype=int)
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
    
    
    def ComputeCriticalPSI(self,PSI):
        """ Function which computes the values of PSI at the:
                - MAGNETIC AXIS ->> PSI_0 
                - SEPARATRIX (LAST CLOSED MAGNETIC SURFACE) / SADDLE POINT ->> PSI_X 
        These values are used to NORMALISE PSI. 
        
        THE METHODOLOGY IS THE FOLLOWING:
            1. OBTAIN CANDIDATE POINTS FOR SOLUTIONS OF EQUATION     NORM(GRAD(PSI))^2 = 0
            2. USING A NEWTON METHOD (OR SOLVER), FIND SOLUTION OF    NORM(GRAD(PSI))^2 = 0
            3. CHECK HESSIAN AT SOLUTIONS TO DIFFERENTIATE BETWEEN EXTREMUM AND SADDLE POINT
            
        THIS IS WHAT WE WOULD DO ANALYTICALLY. IN THE NUMERICAL CASE, WE DO:
            1. INTERPOLATE PSI VALUES ON A FINER STRUCTURED MESH USING PSI ON NODES
            2. COMPUTE GRAD(PSI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
            3. OBTAIN CANDIDATE POINTS ON FINER MESH FOR SOLUTIONS OF EQUATION     NORM(GRAD(PSI))^2 = 0
            4. USING A SOLVER, FIND SOLUTION OF  NORM(GRAD(PSI))^2 = 0   BY EVALUATING AN INTERPOLATION OF GRAD(PSI)
            5. CHECK HESSIAN AT SOLUTIONS
            6. INTERPOLATE VALUE OF PSI AT CRITICAL POINT
        """
        # INTERPOLATION OF GRAD(PSI)
        def gradPSI(X,Rfine,Zfine,gradPSIfine):
            dPSIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPSIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[1].flatten(), (X[0],X[1]), method='cubic')
            GRAD = np.array([dPSIdr,dPSIdz])
            return GRAD
        
        # EVALUATE HESSIAN MATRIX ENTRIES
        def EvaluateHESSIAN(X,gradPSIfine,Rfine,Zfine,dr,dz):
            # compute second derivatives on fine mesh
            dgradPSIdrfine = np.gradient(gradPSIfine[0],dr,dz)
            dgradPSIdzfine = np.gradient(gradPSIfine[1],dr,dz)
            # interpolate HESSIAN components on point 
            dPSIdrdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPSIdzdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[1].flatten(), (X[0],X[1]), method='cubic')
            dPSIdzdz = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdzfine[1].flatten(), (X[0],X[1]), method='cubic')
            if dPSIdrdr*dPSIdzdz-dPSIdzdr**2 > 0:
                return "LOCAL EXTREMUM"
            else:
                return "SADDLE POINT"
            
        
        # 1. INTERPOLATE PSI VALUES ON A FINER STRUCTURED MESH USING PSI ON NODES
        # DEFINE FINER STRUCTURED MESH
        Mr = 60
        Mz = 80
        rfine = np.linspace(self.Xmin, self.Xmax, Mr)
        zfine = np.linspace(self.Ymin, self.Ymax, Mz)
        # INTERPOLATE PSI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine)
        PSIfine = griddata((self.X[:,0],self.X[:,1]), PSI.T[0], (Rfine, Zfine), method='cubic')
        
        # 2. DEFINE GRAD(PSI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
        dr = (rfine[-1]-rfine[0])/Mr
        dz = (zfine[-1]-zfine[0])/Mz
        gradPSIfine = np.gradient(PSIfine,dr,dz)
        
        # FIND SOLUTION OF  GRAD(PSI) = 0   NEAR MAGNETIC AXIS AND SADDLE POINT 
        if self.it == 1:
            self.Xcrit = np.zeros([2,2,3])  # [(iterations n, n+1), (extremum, saddle point), (R_crit,Z_crit,elem_crit)]
            X0_extr = np.array([6,0])
            X0_saddle = np.array([5,-4])
        else:
            X0_extr = self.Xcrit[0,0,:-1]
            X0_saddle = self.Xcrit[0,1,:-1]
            
        # 3. LOOK FOR LOCAL EXTREMUM
        sol = optimize.root(gradPSI, X0_extr, args=(Rfine,Zfine,gradPSIfine))
        if sol.success == True:
            self.Xcrit[1,0,:-1] = sol.x
            # 4. CHECK HESSIAN LOCAL EXTREMUM
            # LOCAL EXTREMUM
            nature = EvaluateHESSIAN(self.Xcrit[1,0,:-1], gradPSIfine, Rfine, Zfine, dr, dz)
            if nature != "LOCAL EXTREMUM":
                print("ERROR IN LOCAL EXTREMUM HESSIAN")
            # 5. INTERPOLATE VALUE OF PSI AT LOCAL EXTREMUM
            # LOOK FOR ELEMENT CONTAINING LOCAL EXTREMUM
            elem = self.SearchElement(self.Xcrit[1,0,:-1],self.PlasmaElems)
            self.Xcrit[1,0,-1] = elem
        else:
            print("LOCAL EXTREMUM NOT FOUND. TAKING PREVIOUS SOLUTION")
            self.Xcrit[1,0,:] = self.Xcrit[0,0,:]
            
        # INTERPOLATE PSI VALUE ON CRITICAL POINT
        self.PSI_0 = self.Elements[int(self.Xcrit[1,0,-1])].ElementalInterpolation(self.Xcrit[1,0,:-1],PSI[self.Elements[int(self.Xcrit[1,0,-1])].Te]) 
        print('LOCAL EXTREMUM AT ',self.Xcrit[1,0,:-1],' (ELEMENT ', int(self.Xcrit[1,0,-1]),') WITH VALUE PSI_0 = ',self.PSI_0)
            
        if self.PLASMA_BOUNDARY == "FREE":
            # 3. LOOK FOR SADDLE POINT
            sol = optimize.root(gradPSI, X0_saddle, args=(Rfine,Zfine,gradPSIfine))
            if sol.success == True:
                self.Xcrit[1,1,:-1] = sol.x 
                # 4. CHECK HESSIAN SADDLE POINT
                nature = EvaluateHESSIAN(self.Xcrit[1,1,:-1], gradPSIfine, Rfine, Zfine, dr, dz)
                if nature != "SADDLE POINT":
                    print("ERROR IN SADDLE POINT HESSIAN")
                # 5. INTERPOLATE VALUE OF PSI AT SADDLE POINT
                # LOOK FOR ELEMENT CONTAINING SADDLE POINT
                elem = self.SearchElement(self.Xcrit[1,1,:-1],np.concatenate((self.VacuumElems,self.PlasmaBoundElems,self.PlasmaElems),axis=0))
                self.Xcrit[1,1,-1] = elem
            else:
                print("SADDLE POINT NOT FOUND. TAKING PREVIOUS SOLUTION")
                self.Xcrit[1,1,:] = self.Xcrit[0,1,:]
                
            # INTERPOLATE PSI VALUE ON CRITICAL POINT
            self.PSI_X = self.Elements[int(self.Xcrit[1,1,-1])].ElementalInterpolation(self.Xcrit[1,1,:-1],PSI[self.Elements[int(self.Xcrit[1,1,-1])].Te]) 
            print('SADDLE POINT AT ',self.Xcrit[1,1,:-1],' (ELEMENT ', int(self.Xcrit[1,1,-1]),') WITH VALUE PSI_X = ',self.PSI_X)
        
        else:
            self.Xcrit[1,1,:-1] = [self.Xmin,self.Ymin]
            self.PSI_X = 0
            
        return 
    
    
    def NormalisePSI(self):
        # NORMALISE SOLUTION OBTAINED FROM SOLVING CutFEM SYSTEM OF EQUATIONS USING CRITICAL PSI VALUES, PSI_0 AND PSI_X
        if self.PLASMA_BOUNDARY == "FREE" or self.PLASMA_CURRENT == "PROFILES":
            for i in range(self.Nn):
                self.PSI_NORM[i,1] = (self.PSI[i]-self.PSI_X)/np.abs(self.PSI_0-self.PSI_X)
        else: 
            for i in range(self.Nn):
                self.PSI_NORM[i,1] = self.PSI[i]
        return 
    
    
    def ComputeTotalPlasmaCurrent(self):
        """ Function that computes de total toroidal current carried by the plasma """            
        return self.PlasmaDomainIntegral(self.Jphi)
    
    def ComputeTotalPlasmaCurrentNormalization(self):
        """ Function that computes the correction factor so that the total current flowing through the plasma region is constant and equal to input file parameter TOTAL_CURRENT. """
        
        # COMPUTE TOTAL PLASMA CURRENT    
        Tcurrent = self.ComputeTotalPlasmaCurrent()
        print('Total plasma current computed = ', Tcurrent)    
        # COMPUTE TOTAL PLASMA CURRENT CORRECTION FACTOR            
        self.gamma = self.TOTAL_CURRENT/Tcurrent
        print("Total plasma current normalization factor = ", self.gamma)
        # COMPUTED NORMALISED TOTAL PLASMA CURRENT
        print("Normalised total plasma current = ", Tcurrent*self.gamma)
        
        return
    
    
    ##################################################################################################
    ###################### CONVERGENCE VALIDATION and VARIABLES UPDATING #############################
    ##################################################################################################
    
    def CheckConvergence(self,VALUES):
        
        if VALUES == "PSI_NORM":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PSI_NORM[:,1]) > 0:
                L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])/np.linalg.norm(self.PSI_NORM[:,1])
            else: 
                L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])
            if L2residu < self.INT_TOL:
                self.converg_INT = True   # STOP INTERNAL WHILE LOOP 
            else:
                self.converg_INT = False
                
            self.residu_INT = L2residu
            print("Internal iteration = ",self.it_INT,", PSI_NORM residu = ", L2residu)
            print(" ")
            
        elif VALUES == "PSI_B":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PSI_B[:,1]) > 0:
                L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])/np.linalg.norm(self.PSI_B[:,1])
            else: 
                L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])
            if L2residu < self.EXT_TOL:
                self.converg_EXT = True   # STOP EXTERNAL WHILE LOOP 
            else:
                self.converg_EXT = False
                
            self.residu_EXT = L2residu
            print("External iteration = ",self.it_EXT,", PSI_B residu = ", L2residu)
            print(" ")
        return 
    
    def UpdatePSI(self,VALUES):
        
        if VALUES == 'PSI_NORM':
            self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
            self.Xcrit[0,:,:] = self.Xcrit[1,:,:]
            
            """if self.converg_INT == False:
                self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
            elif self.converg_INT == True:
                pass"""
        
        elif VALUES == 'PSI_B':
            if self.converg_EXT == False:
                self.PSI_B[:,0] = self.PSI_B[:,1]
                self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
            elif self.converg_EXT == True:
                self.PSI_CONV = self.PSI_NORM[:,1]
        
        return
    
    def UpdateElementalPSI(self):
        """ Function to update the values of PSI_NORM in all mesh elements """
        for element in self.Elements:
            element.PSIe = self.PSI_NORM[element.Te,0]  # TAKE VALUES OF ITERATION N
        return
    
    def UpdateElementalPSI_g(self,INTERFACE):
        """ FUNCTION WHICH UPDATES THE PSI VALUE CONSTRAINTS PSI_g ON ALL INTERFACE EDGES INTEGRATION POINTS. """
        
        if INTERFACE == 'PLASMA/VACUUM':
            for elem in self.PlasmaBoundElems:
                # FOR EACH PLASMA/VACUUM INTERFACE EDGE
                for edge in range(self.Elements[elem].Neint):
                    # ISOLATE ELEMENTAL EDGE
                    EDGE = self.Elements[elem].InterEdges[edge]
                    # COMPUTE INTERFACE CONDITIONS PSI_D
                    EDGE.PSI_g = np.zeros([EDGE.Ngaussint])
                    # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE EDGE
                    for point in range(EDGE.Ngaussint):
                        if self.PLASMA_CURRENT == 'NONLINEAR':
                            EDGE.PSI_g[point] = self.PSIAnalyticalSolution(EDGE.Xgint[point,:],self.PLASMA_CURRENT)
                        elif self.PLASMA_CURRENT == 'PROFILES':
                            EDGE.PSI_g[point] = self.PSI_X
                        else:
                            EDGE.PSI_g[point] = 0
                
        elif INTERFACE == "FIRST WALL":
            # FOR FIXED PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PSI_B = 0, THUS SO ARE ALL ELEMENTAL VALUES  ->> PSI_Be = 0
            if self.PLASMA_BOUNDARY == 'FIXED':  
                for elem in self.VacVessWallElems:
                    for edge in range(self.Elements[elem].Neint):
                        #self.Elements[elem].InterEdges[edge].PSI_g = np.zeros([self.Elements[elem].InterEdges[edge].Ngaussint])
                        # ISOLATE ELEMENTAL EDGE
                        EDGE = self.Elements[elem].InterEdges[edge]
                        # COMPUTE INTERFACE CONDITIONS PSI_D
                        EDGE.PSI_g = np.zeros([EDGE.Ngaussint])
                        # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE EDGE
                        for point in range(EDGE.Ngaussint):
                            EDGE.PSI_g[point] = 0
            
            # FOR FREE PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PSI_B VALUES ARE COMPUTED FROM THE GRAD-SHAFRANOV OPERATOR'S GREEN FUNCTION
            # IN ROUTINE COMPUTEPSI_B, HERE WE NEED TO SEND TO EACH BOUNDARY ELEMENT THE PSI_B VALUES OF THEIR CORRESPONDING BOUNDARY NODES
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
                            EDGE.PSI_g[point] = self.PSI_B[k,0]
                            k += 1
        return
    
    
    ##################################################################################################
    ############################### OPERATIONS OVER GROUPS ###########################################
    ##################################################################################################
    
    def PlasmaDomainIntegral(self,fun):
        """ INTEGRATES FUNCTION fun OVER THE ENTIRE PLASMA DOMAIN.
                fun = fun(X,PSI)        """
        
        integral = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.N @ ELEMENT.PSIe
            # LOOP OVER ELEMENTAL NODES
            for i in range(ELEMENT.n):
                 # LOOP OVER GAUSS NODES
                for ig in range(ELEMENT.Ng2D):
                    integral += fun(ELEMENT.Xg2D[ig,:],PSIg[ig])*ELEMENT.N[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg2D[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.N @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.Ng2D):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            integral += fun(SUBELEM.Xg2D[ig,:],PSIg[ig])*SUBELEM.N[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg2D[ig]            
        return integral
    
    def ComputeActiveNodes(self):
        if self.PLASMA_BOUNDARY == "FIXED":
            plasma_elems = np.concatenate((self.PlasmaBoundElems,self.PlasmaElems), axis=0)
            self.activenodes = set()
            for elem in plasma_elems:
                for node in self.T[elem,:]:
                    self.activenodes.add(node)
            self.activenodes = np.array(list(self.activenodes))
        else:
            self.activenodes = range(self.Nn)
        return
    
    
    ##################### INITIALISATION 
    
    def InitialGuess(self):
        """ This function computes the problem's initial guess, which is taken as the LINEAR CASE SOLUTION WITH SOME RANDOM NOISE. """
        PSI0 = np.zeros([self.Nn])
        if self.PLASMA_CURRENT != 'PROFILES':      
            for i in range(self.Nn):
                #PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.PLASMA_CURRENT)*2*random()
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.PLASMA_CURRENT)
        else:
            for i in range(self.Nn):
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],'LINEAR')*(-0.5)
                
        return PSI0
    
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
            if self.PLASMA_CURRENT != 'ZHENG':
                for i in range(self.Nn):
                    self.PlasmaBoundLevSet[i] = self.PSIAnalyticalSolution(self.X[i,:],'LINEAR')
            else: 
                for i in range(self.Nn):
                    self.PlasmaBoundLevSet[i] = -self.PSIAnalyticalSolution(self.X[i,:],'ZHENG')
                
        elif self.PLASMA_GEOMETRY == "F4E":    # IN THIS CASE, THE PLASMA REGION SHAPE IS DESCRIBED USING THE F4E SHAPE CONTROL POINTS
            self.PlasmaBoundLevSet = self.F4E_PlasmaBoundLevSet()
            
        self.VacVessWallLevSet = (-1)*np.ones([self.Nn])
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            for node_index in self.BoundaryNodes:
                self.VacVessWallLevSet[node_index] = 0
        elif self.VACUUM_VESSEL == "FIRST_WALL":
            for i in range(self.Nn):
                self.VacVessWallLevSet[i] = self.PSIAnalyticalSolution(self.X[i,:],'LINEAR')
            
        return 
    
    def InitialiseElements(self):
        """ Function initialising attribute ELEMENTS which is a list of all elements in the mesh. """
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.PlasmaBoundLevSet[self.T[e,:]],
                                 self.VacVessWallLevSet[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    def InitialisePSI(self):  
        """ INITIALISE PSI VECTORS WHERE THE DIFFERENT SOLUTIONS WILL BE STORED ITERATIVELY DURING THE SIMULATION AND COMPUTE INITIAL GUESS."""
        ####### INITIALISE PSI_NORM UNKNOWN VECTORS
        # INITIALISE PSI VECTORS
        self.PSI = np.zeros([self.Nn])            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORM = np.zeros([self.Nn,2])     # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION FIELD
        
        # COMPUTE INITIAL GUESS AND STORE IT IN INTERNAL SOLUTION FOR N=0
        print('         -> COMPUTE INITIAL GUESS FOR PSI_NORM...', end="")
        self.PSI_NORM[:,0] = self.InitialGuess()  
        # ASSIGN INITIAL GUESS PSI VALUES TO EACH ELEMENT
        self.UpdateElementalPSI()
        print('Done!')   
        
        ####### INITIALISE ELEMENTAL CONSTRAINT VALUES ON PLASMA/VACUUM INTERFACE
        self.PSI_X = 0
        self.UpdateElementalPSI_g('PLASMA/VACUUM')
        
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        if self.PLASMA_CURRENT == "PROFILES":
            #self.gamma = 1
            self.ComputeTotalPlasmaCurrentNormalization()
        
        ####### INITIALISE PSI_B UNKNOWN VECTORS
        # NODES ON VACUUM VESSEL FIRST WALL GEOMETRY FOR WHICH TO EVALUATE PSI_B VALUES == GAUSS INTEGRATION NODES ON FIRST WALL EDGES
        self.NnFW = 0
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                self.NnFW += self.Elements[elem].InterEdges[edge].Ngaussint
        # INITIALISE PSI_B VECTOR
        self.PSI_B = np.zeros([self.NnFW,2])      # VACUUM VESSEL FIRST WALL PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        # FOR VACUUM VESSEL FIRST WALL ELEMENTS, INITIALISE ELEMENTAL ATTRIBUTE PSI_Bg (PSI VALUES ON VACUUM VESSEL FIRST WALL INTERFACE EDGES GAUSS INTEGRATION POINTS)
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                self.Elements[elem].InterEdges[edge].PSI_g = np.zeros([self.Elements[elem].InterEdges[edge].Ngaussint])   
            
        # COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PSI_B 
        print('         -> COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
        self.PSI_B[:,0] = self.ComputePSI_B()
        # ASSIGN INITIAL PSI_B VALUES TO VACUUM VESSEL ELEMENTS
        self.UpdateElementalPSI_g('FIRST WALL')
        print('Done!')    
        
        return
    
    
    def Initialization(self):
        """ Routine which initialises all the necessary elements in the problem """
        
        self.ComputeZhengSolutionCoefficients()
        
        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialiseLevelSets()
        self.writePlasmaBoundaryLS()
        self.writeVacVesselBoundaryLS()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...", end="")
        self.InitialiseElements()
        print('Done!')
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.ClassifyElements()
        self.writeElementsClassification()
        print("Done!")
        
        self.ComputeActiveNodes()

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
        
        # INITIALISE PSI UNKNOWNS
        print("     -> INITIALISE UNKNOWN VECTORS AND COMPUTE INITIAL GUESS...")
        self.InitialisePSI()
        self.writePSI()
        self.writePSI_NORM()
        self.writePSI_B()
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
        self.XIg1D, self.Wg1D, self.Ng1D = GaussQuadrature(0,self.QuadratureOrder)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        #### QUADRATURE TO INTEGRATE LINES (1D)
        self.N1D, self.dNdxi1D, foo = EvaluateReferenceShapeFunctions(self.XIg1D, 0, self.nsole-1, self.nsole)
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
                # . RECALL THAT PLASMA REGION IS DEFINED BY NEGATIVE VALUES OF LEVEL-SET -> NEED TO INVERT SIGN
                # . CLOSED GEOMETRY DEFINED BY 0-LEVEL CONTOUR BENEATH ACTIVE SADDLE POINT (DIVERTOR REGION) NEEDS TO BE
                #   DISCARTED BECAUSE THE LEVEL-SET DESCRIBES ONLY THE PLASMA REGION GEOMETRY -> NEED TO POST-PROCESS CUTFEM
                #   SOLUTION IN ORDER TO TAKE ITS 0-LEVEL CONTOUR ENCLOSING ONLY THE PLASMA REGION. 
                
                # 1. INVERT SIGN DEPENDING ON SOLUTION PLASMA REGION SIGN
                if self.PSI_0 > 0: # WHEN THE OBTAINED SOLUTION IS POSITIVE INSIDE THE PLASMA
                    self.PlasmaBoundLevSet = -self.PSI_NORM[:,1].copy()
                else: # WHEN THE OBTAINED SOLUTION IS NEGATIVE INSIDE THE PLASMA
                    self.PlasmaBoundLevSet = self.PSI_NORM[:,1].copy() 
                    
                # 2. DISCARD DIVERTOR ENCLOSED REGION (BENEATH ACTIVE SADDLE POINT)
                Zlow = self.Xcrit[1,1,1] - 0.05
                for i in range(self.Nn):  
                    if self.X[i,1] < Zlow:
                        self.PlasmaBoundLevSet[i] = np.abs(self.PlasmaBoundLevSet[i])
                        
                # 3. DISCARD POSSIBLE EXISTING ENCLOSED REGIONS AT LEFT-HAND-SIDE FROM PLASMA REGION
                # OBTAIN LEFTMOST POINT FROM PLASMA REGION SEPARATRIX
                fig, ax = plt.subplots(figsize=(6, 8))
                cs = ax.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0])
                for item in cs.collections:
                    for i in item.get_paths():
                        v = i.vertices
                        x_equ = v[:, 0]+0.1
                        y_equ = v[:, 1]+0.1
                fig.clear()
                plt.close(fig)
                
                Rleft = np.min(x_equ) - 0.2
                for i in range(self.Nn):  
                    if self.X[i,0] < Rleft:  # DISCARD LEVEL-SET REGION LEFT FROM LEFTMOST POINT
                        self.PlasmaBoundLevSet[i] = np.abs(self.PlasmaBoundLevSet[i])
                        
                #self.PlotLevelSetEvolution(Zlow,Rleft)

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
        
        # OPEN ELEMENTAL MATRICES OUTPUT FILE
        if self.ELMAT_output:
            self.ELMAT_file = open('ElementalMatrices.dat', 'w')
            self.ELMAT_file.write('ELEMENTAL_MATRICES_FILE\n')
            self.ELMAT_file.write('NON_CUT_ELEMENTS\n')
        
        # INTEGRATE OVER THE SURFACE OF ELEMENTS WHICH ARE NOT CUT BY ANY INTERFACE (STANDARD QUADRATURES)
        print("     Integrate over non-cut elements...", end="")
        
        for elem in self.NonCutElems: 
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]  
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.Ng2D])
            if ELEMENT.Dom < 0:
                # MAP PSI VALUES FROM ELEMENT NODES TO GAUSS NODES
                PSIg = ELEMENT.N @ ELEMENT.PSIe
                for ig in range(self.Elements[elem].Ng2D):
                    if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                        Xg = ELEMENT.Xg2D[ig,:]/self.R0
                    else:
                        Xg = ELEMENT.Xg2D[ig,:]
                    SourceTermg[ig] = self.mu0*Xg[0]*self.Jphi(Xg,PSIg[ig])
                    
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg,self.R0)
            else:
                LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg)
            
            if self.ELMAT_output:
                self.ELMAT_file.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.ELMAT_file.write('elmat\n')
                np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                self.ELMAT_file.write('elrhs\n')
                np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(ELEMENT.n):   # ROWS ELEMENTAL MATRIX
                for j in range(ELEMENT.n):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_NON_CUT_ELEMENTS\n')
            self.ELMAT_file.write('CUT_ELEMENTS_SURFACE\n')
        
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
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.N @ ELEMENT.PSIe
                    for ig in range(SUBELEM.Ng2D):
                        if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                            Xg = SUBELEM.Xg2D[ig,:]/self.R0
                        else:
                            Xg = SUBELEM.Xg2D[ig,:]
                        SourceTermg[ig] = self.mu0*Xg[0]*self.Jphi(Xg,PSIg[ig])
                        
                # COMPUTE ELEMENTAL MATRICES
                if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                    LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg,self.R0)
                else:
                    LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg)
                
                if self.ELMAT_output:
                    self.ELMAT_file.write("elem {:d} {:d} subelem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom,SUBELEM.index,SUBELEM.Dom))
                    self.ELMAT_file.write('elmat\n')
                    np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                    self.ELMAT_file.write('elrhs\n')
                    np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
                
                # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
                for i in range(len(SUBELEM.Te)):   # ROWS ELEMENTAL MATRIX
                    for j in range(len(SUBELEM.Te)):   # COLUMNS ELEMENTAL MATRIX
                        self.LHS[SUBELEM.Te[i],SUBELEM.Te[j]] += LHSe[i,j]
                    self.RHS[SUBELEM.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_CUT_ELEMENTS_SURFACE\n')
            self.ELMAT_file.write('CUT_ELEMENTS_INTERFACE\n')
        
        # INTEGRATE OVER THE CUT EDGES IN ELEMENTS CUT BY INTERFACES (MODIFIED QUADRATURES)
        print("     Integrate along cut-elements interface edges...", end="")
        
        for elem in self.CutElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[elem]
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta,self.R0)
            else: 
                LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta)
                
            if self.ELMAT_output:
                self.ELMAT_file.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.ELMAT_file.write('elmat\n')
                np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                self.ELMAT_file.write('elrhs\n')
                np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(len(ELEMENT.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(ELEMENT.Te)):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
                
        if self.ELMAT_output:
            self.ELMAT_file.write('END_CUT_ELEMENTS_INTERFACE\n')
        
        # IN THE CASE WHERE THE VACUUM VESSEL FIRST WALL CORRESPONDS TO THE COMPUTATIONAL DOMAIN'S BOUNDARY, ELEMENTS CONTAINING THE FIRST WALL ARE NOT CUT ELEMENTS 
        # BUT STILL WE NEED TO INTEGRATE ALONG THE COMPUTATIONAL DOMAIN'S BOUNDARY BOUNDARY 
        
        if self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
            
            if self.ELMAT_output:
                self.ELMAT_file.write('BOUNDARY_ELEMENTS_INTERFACE\n')
            
            for elem in self.VacVessWallElems:
                # ISOLATE ELEMENT 
                ELEMENT = self.Elements[elem]
                # COMPUTE ELEMENTAL MATRICES
                if self.PLASMA_CURRENT == "LINEAR" or self.PLASMA_CURRENT == "NONLINEAR":  # DIMENSIONLESS SOLUTION CASE
                    LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta,self.R0)
                else:
                    LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta)
                
                if self.ELMAT_output:
                    self.ELMAT_file.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                    self.ELMAT_file.write('elmat\n')
                    np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                    self.ELMAT_file.write('elrhs\n')
                    np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
                
                # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
                for i in range(len(ELEMENT.Te)):   # ROWS ELEMENTAL MATRIX
                    for j in range(len(ELEMENT.Te)):   # COLUMNS ELEMENTAL MATRIX
                        self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                    self.RHS[ELEMENT.Te[i]] += RHSe[i]
                    
            if self.ELMAT_output:
                self.ELMAT_file.write('END_BOUNDARY_ELEMENTS_INTERFACE\n')
            
        if self.ELMAT_output:
            self.ELMAT_file.write('END_ELEMENTAL_MATRICES_FILE\n')
            self.ELMAT_file.close()
                
        print("Done!") 
        
        if self.GlobalSystem_output:
            with open('GlobalSystem.dat', 'w') as GlobalSystemfile:
                GlobalSystemfile.write('LHS\n')
                for i in range(self.Nn):
                    for j in range(self.Nn):
                        if self.LHS[i,j] != 0.0:
                            GlobalSystemfile.write("{:d} {:d} {:e}\n".format(i,j,self.LHS[i,j]))
                GlobalSystemfile.write('RHS\n')
                for i in range(self.Nn):
                    GlobalSystemfile.write("{:e}\n".format(self.RHS[i,0]))
        
        return
    
    def SolveSystem(self):
        # SOLVE LINEAR SYSTEM OF EQUATIONS AND OBTAIN PSI
        self.PSI = np.linalg.solve(self.LHS, self.RHS)
        return
    
    def StorePSIValues(self):
        self.PSI_NORM_ALL[:,self.it] = self.PSI_NORM[:,0]   
        if self.it > 0:
            self.PSI_crit_ALL[0,0,self.it] = self.PSI_0
            self.PSI_crit_ALL[0,1:,self.it] = self.Xcrit[1,0,:-1]
            self.PSI_crit_ALL[1,0,self.it] = self.PSI_X
            self.PSI_crit_ALL[1,1:,self.it] = self.Xcrit[1,1,:-1]
        return
    
    def StoreMeshConfiguration(self):
        self.PlasmaBoundLevSet_ALL[:,self.it] = self.PlasmaBoundLevSet
        self.ElementalGroups_ALL[:,self.it] = self.ObtainClassification()
        return
    
    ##################################################################################################
    ############################################# OUTPUT #############################################
    ##################################################################################################
    
    def openOUTPUTfiles(self):
        
        self.PSI_file = open(self.outputdir+'/UNKNO.dat', 'w')
        self.PSI_file.write('UNKNOWN_PSIpol_FIELD\n')
        
        self.PSI_NORM_file = open(self.outputdir+'/PSIpol.dat', 'w')
        self.PSI_NORM_file.write('PSIpol_FIELD\n')
        
        self.PSIcrit_file = open(self.outputdir+'/PSIcrit.dat', 'w')
        self.PSIcrit_file.write('PSIcrit_VALUES\n')
        
        self.PSI_B_file = open(self.outputdir+'/PSIpol_B.dat', 'w')
        self.PSI_B_file.write('PSIpol_B_VALUES\n')
        
        self.RESIDU_file = open(self.outputdir+'/Residu.dat', 'w')
        self.RESIDU_file.write('RESIDU_VALUES\n')
        
        self.ElementsClassi_file = open(self.outputdir+'/MeshElementsClassification.dat', 'w')
        self.ElementsClassi_file.write('MESH_ELEMENTS_CLASSIFICATION\n')
        
        self.PlasmaLevSetVals_file = open(self.outputdir+'/PlasmaBoundLS.dat', 'w')
        self.PlasmaLevSetVals_file.write('PLASMA_BOUNDARY_LEVEL_SET\n')
        
        self.VacVessLevSetVals_file = open(self.outputdir+'/VacuumVesselWallLS.dat', 'w')
        self.VacVessLevSetVals_file.write('VACUUM_VESSEL_LEVEL_SET\n')
        
        return
    
    def closeOUTPUTfiles(self):
        self.PSI_file.write('END_UNKNOWN_PSIpol_FIELD')
        self.PSI_file.close()
        
        self.PSI_NORM_file.write('END_PSIpol_FIELD')
        self.PSI_NORM_file.close()
        
        self.PSIcrit_file.write('END_PSIcrit_VALUES')
        self.PSIcrit_file.close()
        
        self.PSI_B_file.write('END_PSIpol_B_VALUES')
        self.PSI_B_file.close()
        
        self.RESIDU_file.write('END_RESIDU_VALUES')
        self.RESIDU_file.close()
        
        self.ElementsClassi_file.write('END_MESH_ELEMENTS_CLASSIFICATION')
        self.ElementsClassi_file.close()
        
        self.VacVessLevSetVals_file.write('END_VACUUM_VESSEL_LEVEL_SET')
        self.VacVessLevSetVals_file.close()
        
        self.PlasmaLevSetVals_file.write('END_PLASMA_BOUNDARY_LEVEL_SET')
        self.PlasmaLevSetVals_file.close()
        return
    
    def copysimfiles(self):
        
        # COPY DOM.DAT FILE
        MeshDataFile = self.mesh_folder +'/'+ 'TS-CUTFEM-' + self.MESH +'.dom.dat'
        shutil.copy2(MeshDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.dom.dat')
        # COPY GEO.DAT FILE
        MeshFile = self.mesh_folder +'/'+ 'TS-CUTFEM-' + self.MESH +'.geo.dat'
        shutil.copy2(MeshFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.geo.dat')
        # COPY EQU.DAT FILE
        EQUILIDataFile = self.case_file +'.equ.dat'
        shutil.copy2(EQUILIDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.equ.dat')
        
        return
    
    def writeparams(self):
        
        self.PARAMS_file = open(self.outputdir+'/PARAMETERS.dat', 'w')
        self.PARAMS_file.write('SIMULATION_PARAMTERS_FILE\n')
        self.PARAMS_file.write('\n')
        
        self.PARAMS_file.write('MESH_PARAMETERS\n')
        self.PARAMS_file.write("    NPOIN = {:d}\n".format(self.Nn))
        self.PARAMS_file.write("    NELEM = {:d}\n".format(self.Ne))
        self.PARAMS_file.write("    ELEM = {:d}\n".format(self.ElType))
        self.PARAMS_file.write("    NBOUN = {:d}\n".format(self.Nbound))
        self.PARAMS_file.write('END_MESH_PARAMETERS\n')
        self.PARAMS_file.write('\n')
        
        self.PARAMS_file.write('PROBLEM_TYPE_PARAMETERS\n')
        self.PARAMS_file.write("    PLASMA_BOUNDARY_equ = " + self.PLASMA_BOUNDARY)
        self.PARAMS_file.write("    PLASMA_GEOMETRY_equ = " + self.PLASMA_GEOMETRY)
        self.PARAMS_file.write("    PLASMA_CURRENT_equ = " + self.PLASMA_CURRENT)
        self.PARAMS_file.write("    VACUUM_VESSEL_equ = " + self.VACUUM_VESSEL)
        self.PARAMS_file.write("    TOTAL_PLASMA_CURRENT = {:f}\n".format(self.TOTAL_CURRENT))
        self.PARAMS_file.write('END_PROBLEM_TYPE_PARAMETERS\n')
        self.PARAMS_file.write('\n')
        
        self.PARAMS_file.write('VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
        self.PARAMS_file.write("    R_MAX_equ = {:f}\n".format(self.Rmax))
        self.PARAMS_file.write("    R_MIN_equ = {:f}\n".format(self.Rmin))
        self.PARAMS_file.write("    EPSILON_equ = {:f}\n".format(self.epsilon))
        self.PARAMS_file.write("    KAPPA_equ = {:f}\n".format(self.kappa))
        self.PARAMS_file.write("    DELTA_equ = {:f}\n".format(self.delta))
        self.PARAMS_file.write('END_VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
        self.PARAMS_file.write('\n')
        
        if self.PLASMA_GEOMETRY == 'F4E':
            self.PARAMS_file.write('PLASMA_REGION_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write("    CONTROL_POINTS_equ = {:d}\n".format(self.CONTROL_POINTS))
            self.PARAMS_file.write("    R_SADDLE_equ = {:f}\n".format(self.R_SADDLE))
            self.PARAMS_file.write("    Z_SADDLE_equ = {:f}\n".format(self.Z_SADDLE))
            self.PARAMS_file.write("    R_RIGHTMOST_equ = {:f}\n".format(self.R_RIGHTMOST))
            self.PARAMS_file.write("    Z_RIGHTMOST_equ = {:f}\n".format(self.Z_RIGHTMOST))
            self.PARAMS_file.write("    R_LEFTMOST_equ = {:f}\n".format(self.R_LEFTMOST))
            self.PARAMS_file.write("    Z_LEFTMOST_equ = {:f}\n".format(self.Z_LEFTMOST))
            self.PARAMS_file.write("    R_TOP_equ = {:f}\n".format(self.R_TOP))
            self.PARAMS_file.write("    Z_TOP_equ = {:f}\n".format(self.Z_TOP))
            self.PARAMS_file.write('END_PLASMA_REGION_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write('\n')
        
        if self.PLASMA_CURRENT == 'PROFILES':
            self.PARAMS_file.write('PLASMA_CURRENT_MODEL_PARAMETERS\n')
            self.PARAMS_file.write("    B0_equ = {:f}\n".format(self.B0))
            self.PARAMS_file.write("    q0_equ = {:f}\n".format(self.q0))
            self.PARAMS_file.write("    n_p_equ = {:f}\n".format(self.n_p))
            self.PARAMS_file.write("    g0_equ = {:f}\n".format(self.G0))
            self.PARAMS_file.write("    n_g_equ = {:f}\n".format(self.n_g))
            self.PARAMS_file.write('END_PLASMA_CURRENT_MODEL_PARAMETERS\n')
            self.PARAMS_file.write('\n')
        
        if self.PLASMA_BOUNDARY == 'FREE':
            self.PARAMS_file.write('EXTERNAL_COILS_PARAMETERS\n')
            self.PARAMS_file.write("    N_COILS_equ = {:d}\n".format(self.Ncoils))
            for icoil in range(self.Ncoils):
                self.PARAMS_file.write("    Rposi = {:f}\n".format(self.Xcoils[icoil,0]))
                self.PARAMS_file.write("    Zposi = {:f}\n".format(self.Xcoils[icoil,1]))
                self.PARAMS_file.write("    Inten = {:f}\n".format(self.Icoils[icoil]))
                self.PARAMS_file.write('\n')
            self.PARAMS_file.write('END_EXTERNAL_COILS_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('EXTERNAL_SOLENOIDS_PARAMETERS\n')
            self.PARAMS_file.write("    N_SOLENOIDS_equ = {:d}\n".format(self.Nsolenoids))
            for icoil in range(self.Ncoils):
                self.PARAMS_file.write("    Rposi = {:f}\n".format(self.Xcoils[icoil,0]))
                self.PARAMS_file.write("    Zposi = {:f}\n".format(self.Xcoils[icoil,1]))
                self.PARAMS_file.write("    Inten = {:f}\n".format(self.Icoils[icoil]))
                self.PARAMS_file.write('\n')
            self.PARAMS_file.write('END_EXTERNAL_SOLENOIDS_PARAMETERS\n')
            self.PARAMS_file.write('\n')
        
        self.PARAMS_file.write('NUMERICAL_TREATMENT_PARAMETERS\n')
        self.PARAMS_file.write("    QUADRATURE_ORDER_equ = {:d}\n".format(self.QuadratureOrder))
        self.PARAMS_file.write("    MAX_EXT_IT_equ = {:d}\n".format(self.EXT_ITER))
        self.PARAMS_file.write("    EXT_TOL_equ = {:e}\n".format(self.EXT_TOL))
        self.PARAMS_file.write("    MAX_INT_IT_equ = {:d}\n".format(self.INT_ITER))
        self.PARAMS_file.write("    INT_TOL_equ = {:e}\n".format(self.INT_TOL))
        self.PARAMS_file.write('END_NUMERICAL_TREATMENT_PARAMETERS\n')
        self.PARAMS_file.write('\n')
        
        self.PARAMS_file.write('END_SIMULATION_PARAMTERS_FILE\n')
        self.PARAMS_file.close()
        return
    
    def writePSI(self):
        self.PSI_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Nn):
            self.PSI_file.write("{:d} {:e}\n".format(inode+1,float(self.PSI[inode])))
        self.PSI_file.write('END_ITERATION\n')
        return
    
    def writePSI_NORM(self):
        self.PSI_NORM_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Nn):
            self.PSI_NORM_file.write("{:d} {:e}\n".format(inode+1,self.PSI_NORM[inode,0]))
        self.PSI_NORM_file.write('END_ITERATION\n')
        return
    
    def writePSI_B(self):
        self.PSI_B_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.NnFW):
            self.PSI_B_file.write("{:d} {:e}\n".format(inode+1,self.PSI_B[inode,0]))
        self.PSI_B_file.write('END_ITERATION\n')
        return
    
    def writeresidu(self,which_loop):
        
        if which_loop == "INTERNAL":
            if self.it_INT == 1:
                self.RESIDU_file.write("INTERNAL_LOOP_STRUCTURE\n")
            self.RESIDU_file.write("  INTERNAL_ITERATION = {:d} \n".format(self.it_INT))
            self.RESIDU_file.write("      INTERNAL_RESIDU = {:f} \n".format(self.residu_INT))
            
        elif which_loop == "EXTERNAL":
            self.RESIDU_file.write("END_INTERNAL_LOOP_STRUCTURE\n")
            self.RESIDU_file.write("EXTERNAL_ITERATION = {:d} \n".format(self.it_EXT))
            self.RESIDU_file.write("  EXTERNAL_RESIDU = {:f} \n".format(self.residu_EXT))
        
        return
    
    def writePSIcrit(self):
        self.PSIcrit_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}".format(self.Xcrit[1,0,-1],self.Xcrit[1,0,0],self.Xcrit[1,0,1],self.PSI_0))
        self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}".format(self.Xcrit[1,1,-1],self.Xcrit[1,1,0],self.Xcrit[1,1,1],self.PSI_X))
        self.PSIcrit_file.write('END_ITERATION\n')
        return
    
    def writeElementsClassification(self):
        self.ElementsClassi_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        MeshClassi = self.ObtainClassification()
        for ielem in range(self.Ne):
            self.ElementsClassi_file.write("{:d} {:d}\n".format(ielem+1,MeshClassi[ielem]))
        self.ElementsClassi_file.write('END_ITERATION\n')
        return
    
    def writePlasmaBoundaryLS(self):
        self.PlasmaLevSetVals_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Nn):
            self.PlasmaLevSetVals_file.write("{:d} {:e}\n".format(inode+1,self.PlasmaBoundLevSet[inode]))
        self.PlasmaLevSetVals_file.write('END_ITERATION\n')
        return
    
    def writeVacVesselBoundaryLS(self):
        for inode in range(self.Nn):
            self.VacVessLevSetVals_file.write("{:d} {:e}\n".format(inode+1,self.VacVessWallLevSet[inode]))
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
        
        #self.output_file = open('OUTPUT.dat', 'w')
        
        # OUTPUT RESULTS FOLDER
        # Check if the directory exists
        if not os.path.exists(self.outputdir):
            # Create the directory
            os.makedirs(self.outputdir)
            
        self.copysimfiles()
        
        self.writeparams()
        self.openOUTPUTfiles()    
                
        # INITIALIZATION
        print("INITIALIZATION...")
        self.it = 0
        self.it_EXT = 0
        self.it_INT = 0
        self.Initialization()
        print('Done!')

        self.PlotSolution(self.PSI_NORM[:,0])  # PLOT INITIAL SOLUTION

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.converg_EXT = False
        self.it_EXT = 0
        # STORE INITIAL VALUES
        #self.StorePSIValues()
        #self.StoreMeshConfiguration()
        while (self.converg_EXT == False and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = False
            self.it_INT = 0
            #****************************************
            while (self.converg_INT == False and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                #self.StoreMeshConfiguration()
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                
                # WRITE ITERATION CONFIGURATION
                self.writePlasmaBoundaryLS()
                self.writeElementsClassification()
                
                ##################################
                #self.PlotClassifiedElements_2()
                # COMPUTE TOTAL PLASMA CURRENT CORRECTION FACTOR
                if self.PLASMA_CURRENT == "PROFILES":
                    Tcurrent = self.ComputeTotalPlasmaCurrent()
                    print("Total plasma current = ", Tcurrent)
                #self.PlotPROFILES()
                #self.PlotJphi_JphiNORM()
                self.AssembleGlobalSystem() 
                self.SolveSystem()                          # 1. SOLVE CutFEM SYSTEM  ->> PSI
                self.writePSI()
                self.ComputeCriticalPSI(self.PSI)           # 2. COMPUTE CRITICAL VALUES   PSI_0 AND PSI_X
                #self.writePSIcrit()
                self.NormalisePSI()                         # 3. NORMALISE PSI RESPECT TO CRITICAL VALUES  ->> PSI_NORM 
                self.writePSI_NORM()
                self.PlotPSI_PSINORM()
                self.CheckConvergence('PSI_NORM')           # 4. CHECK CONVERGENCE OF PSI_NORM FIELD
                self.writeresidu("INTERNAL")
                self.UpdateElements()                       # 5. UPDATE MESH ELEMENTS CLASSIFACTION RESPECT TO NEW PLASMA BOUNDARY LEVEL-SET
                self.UpdatePSI('PSI_NORM')                  # 6. UPDATE PSI_NORM VALUES (PSI_NORM[:,0] = PSI_NORM[:,1])
                self.UpdateElementalPSI()                   # 7. UPDATE PSI_NORM VALUES IN CORRESPONDING ELEMENTS (ELEMENT.PSIe = PSI_NORM[ELEMENT.Te,0])
                self.UpdateElementalPSI_g("PLASMA/VACUUM")  # 8. UPDATE ELEMENTAL CONSTRAINT VALUES PSI_g FOR PLASMA/VACUUM INTERFACE
                #self.StorePSIValues()
                
                ##################################
            self.ComputeTotalPlasmaCurrentNormalization()
            print('COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
            self.PSI_B[:,1] = self.ComputePSI_B()     # COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B WITH INTERNALLY CONVERGED PSI_NORM
            print('Done!')
            self.CheckConvergence('PSI_B')            # CHECK CONVERGENCE OF VACUUM VESSEL FIEST WALL PSI VALUES  (PSI_B)
            self.writeresidu("EXTERNAL")
            self.UpdatePSI('PSI_B')                   # UPDATE PSI_NORM AND PSI_B VALUES
            self.UpdateElementalPSI_g("FIRST WALL")   # UPDATE ELEMENTAL CONSTRAINT VALUES PSI_g FOR VACUUM VESSEL FIRST WALL INTERFACE 
            #****************************************
        print('SOLUTION CONVERGED')
        #self.PlotSolution(self.PSI_CONV)
        
        self.closeOUTPUTfiles()
        
        return
    
    
    ##################################################################################################
    ############################### RENDERING AND REPRESENTATION #####################################
    ##################################################################################################
    
    def PlotSolution(self,PSI):
        
        if len(np.shape(PSI)) == 2:
            PSI = PSI[:,0]
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        axs.set_xlim(self.Xmin,self.Xmax)
        axs.set_ylim(self.Ymin,self.Ymax)
        a = axs.tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], PSI[self.activenodes], levels=30)
        axs.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs.tricontour(self.X[self.activenodes,0],self.X[self.activenodes,1], PSI[self.activenodes], levels=[0], colors = 'black')
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
            Jphi[i] = self.Jphi(self.X[i,:],self.PSI_NORM[i,0])
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
    
    
    def PlotPSI_PSINORM(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PSI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, AND PSI_NORM (NORMALISED PSI). """
        
        fig, axs = plt.subplots(1, 2, figsize=(11,5))
        for i in range(2):
            axs[i].set_xlim(self.Xmin, self.Xmax)
            axs[i].set_ylim(self.Ymin, self.Ymax)
        
        # CENTRAL PLOT: PSI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
        a0 = axs[0].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], self.PSI[self.activenodes,0], levels=50)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI[:,0], levels=[0], colors = 'black')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].set_title('POLOIDAL MAGNETIC FLUX PSI')
        plt.colorbar(a0, ax=axs[0])
        
        # RIGHT PLOT: PSI at iteration N+1 WITH NORMALISATION
        a1 = axs[1].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], self.PSI_NORM[self.activenodes,1], levels=50)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[1].set_title('NORMALIZED POLOIDAL MAGNETIC FLUX PSI_NORM')
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
    
    
    def InspectElement(self,element_index,PSI,INTERFACE,QUADRATURE):
        
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
        if PSI:
            axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=30, cmap='plasma')
            axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        # PLOT ELEMENT EDGES
        for iedge in range(numedges):
            axs[0].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=3)

        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=3)
        axs[1].scatter(ELEM.Xe[:,0],ELEM.Xe[:,1], marker='o', s=70, zorder=5)
        if INTERFACE:
            for iedge in range(ELEM.Neint):
                axs[1].scatter(ELEM.InterEdges[iedge].Xeint[:,0],ELEM.InterEdges[iedge].Xeint[:,1],marker='.',color='red',s=50, zorder=5)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg2D[:,0],ELEM.Xg2D[:,1],marker='x',c='black')
            elif ELEM.Dom == 2 and self.VACUUM_VESSEL == "COMPUTATIONAL_DOMAIN":
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg2D[:,0],ELEM.Xg2D[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for iedge in range(ELEM.Neint):
                    axs[1].scatter(ELEM.InterEdges[iedge].Xgint[:,0],ELEM.InterEdges[iedge].Xgint[:,1],marker='x',color='grey',s=50, zorder = 5)
            else:
                for SUBELEM in ELEM.SubElements:
                    # PLOT SUBELEMENT EDGES
                    for i in range(numedges):
                        plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='black', linewidth=1)
                    # PLOT QUADRATURE INTEGRATION POINTS
                    plt.scatter(SUBELEM.Xg2D[:,0],SUBELEM.Xg2D[:,1],marker='x',c='grey', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for iedge in range(ELEM.Neint):
                    axs[1].scatter(ELEM.InterEdges[iedge].Xgint[:,0],ELEM.InterEdges[iedge].Xgint[:,1],marker='x',color='grey',s=50, zorder=5)
        return
    
    
    def PlotLevelSetEvolution(self,Zlow,Rleft):
        
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].set_xlim(self.Xmin,self.Xmax)
        axs[0].set_ylim(self.Ymin,self.Ymax)
        a = axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=30)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Xmin,self.Xmax)
        axs[1].set_ylim(self.Ymin,self.Ymax)
        a = axs[1].tricontourf(self.X[:,0],self.X[:,1], np.sign(self.PlasmaBoundLevSet), levels=30)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black',linewidths = 3)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linestyles = 'dashed')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet_ALL[:,self.it-1], levels=[0], colors = 'orange',linestyles = 'dashed')
        axs[1].plot([self.Xmin,self.Xmax],[Zlow,Zlow],color = 'green')
        axs[1].plot([Rleft,Rleft],[self.Ymin,self.Ymax],color = 'green')

        plt.show()
        
        return
    
    
    def PlotMesh(self):
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        # Plot nodes
        plt.plot(self.X[:,0],self.X[:,1],'.')
        # Plot element edges
        for e in range(self.Ne):
            for i in range(self.numvertices):
                plt.plot([self.X[self.T[e,i],0], self.X[self.T[e,int((i+1)%self.n)],0]], 
                        [self.X[self.T[e,i],1], self.X[self.T[e,int((i+1)%self.n)],1]], color='black', linewidth=1)
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
    
    
    # PREPARE FUNCTION WHICH PLOTS AT THE SAME TIME PSI_Dg and PSI_Bg, that is plots PSI_g
    def PlotInterfaceValues(self):
        """ Function which plots the values PSI_g at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """

        # COLLECT PSI_g DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D,self.dim])
        PSI_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PSI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for elem in self.PlasmaBoundElems:
            for edge in range(self.Elements[elem].Neint):
                for point in range(self.Elements[elem].InterEdges[edge].Ngaussint):
                    X_Dg[k,:] = self.Elements[elem].InterEdges[edge].Xgint[point]
                    PSI_Dg[k] = self.Elements[elem].InterEdges[edge].PSI_g[point]
                    k += 1
            for node in range(self.Elements[elem].n):
                X_D[l,:] = self.Elements[elem].Xe[node,:]
                PSI_D[l] = self.PSI[self.Elements[elem].Te[node]]
                l += 1
            
        # COLLECT PSI_g DATA ON VACUUM VESSEL FIRST WALL 
        X_Bg = np.zeros([self.NnFW,self.dim])
        PSI_Bg = np.zeros([self.NnFW])
        X_B = np.zeros([len(self.VacVessWallElems)*self.n,self.dim])
        PSI_B = np.zeros([len(self.VacVessWallElems)*self.n])
        k = 0
        l = 0
        for elem in self.VacVessWallElems:
            for edge in range(self.Elements[elem].Neint):
                for point in range(self.Elements[elem].InterEdges[edge].Ngaussint):
                    X_Bg[k,:] = self.Elements[elem].InterEdges[edge].Xgint[point]
                    PSI_Bg[k] = self.Elements[elem].InterEdges[edge].PSI_g[point]
                    k += 1
            for node in range(self.Elements[elem].n):
                X_B[l,:] = self.Elements[elem].Xe[node,:]
                PSI_B[l] = self.PSI[self.Elements[elem].Te[node]]
                l += 1
            
        fig, axs = plt.subplots(1, 2, figsize=(14,7))
        ### UPPER ROW SUBPLOTS 
        # LEFT SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[0].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        cmap = plt.get_cmap('jet')
        
        norm = plt.Normalize(np.min([PSI_Bg.min(),PSI_Dg.min()]),np.max([PSI_Bg.max(),PSI_Dg.max()]))
        linecolors_Dg = cmap(norm(PSI_Dg))
        linecolors_Bg = cmap(norm(PSI_Bg))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)
        axs[0].scatter(X_Bg[:,0],X_Bg[:,1],color = linecolors_Bg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[1].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        linecolors_B = cmap(norm(PSI_B))
        axs[1].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        axs[1].scatter(X_B[:,0],X_B[:,1],color = linecolors_B)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])

        plt.show()
        return
    
    
    def PlotPlasmaBoundaryConstraints(self):
        
        # COLLECT PSI_g DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D,self.dim])
        PSI_Dexact = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        PSI_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PSI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for elem in self.PlasmaBoundElems:
            for edge in range(self.Elements[elem].Neint):
                for point in range(self.Elements[elem].InterEdges[edge].Ngaussint):
                    X_Dg[k,:] = self.Elements[elem].InterEdges[edge].Xgint[point]
                    if self.PLASMA_CURRENT != 'PROFILES':
                        PSI_Dexact[k] = self.PSIAnalyticalSolution(X_Dg[k,:],self.PLASMA_CURRENT)
                    else:
                        PSI_Dexact[k] = self.Elements[elem].InterEdges[edge].PSI_g[point]
                    PSI_Dg[k] = self.Elements[elem].InterEdges[edge].PSI_g[point]
                    k += 1
            for node in range(self.Elements[elem].n):
                X_D[l,:] = self.Elements[elem].Xe[node,:]
                PSI_D[l] = self.PSI[self.Elements[elem].Te[node]]
                l += 1
            
        fig, axs = plt.subplots(1, 3, figsize=(18,6)) 
        # LEFT SUBPLOT: ANALYTICAL VALUES
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[0].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PSI_Dexact.min(),PSI_Dexact.max())
        linecolors_Dexact = cmap(norm(PSI_Dexact))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dexact)
        
        # CENTER SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[1].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        #norm = plt.Normalize(PSI_Dg.min(),PSI_Dg.max())
        linecolors_Dg = cmap(norm(PSI_Dg))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[2].set_aspect('equal')
        axs[2].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[2].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        axs[2].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[2])

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

    
    def PlotError(self):
        
        AnaliticalNorm = np.zeros([len(self.activenodes)])
        error = np.zeros([len(self.activenodes)])
        for inode in range(len(self.activenodes)):
            AnaliticalNorm[inode] = self.PSIAnalyticalSolution(self.X[self.activenodes[inode],:],self.PLASMA_CURRENT)
            error[inode] = abs(AnaliticalNorm[inode]-self.PSI_CONV[self.activenodes[inode]])
            
        print(np.linalg.norm(error))
            
        fig, axs = plt.subplots(1, 3, figsize=(16,5))
        axs[0].set_xlim(self.Xmin,self.Xmax)
        axs[0].set_ylim(self.Ymin,self.Ymax)
        a = axs[0].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], AnaliticalNorm, levels=30)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].tricontour(self.X[self.activenodes,0],self.X[self.activenodes,1], AnaliticalNorm, levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Xmin,self.Xmax)
        axs[1].set_ylim(self.Ymin,self.Ymax)
        a = axs[1].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], self.PSI_CONV[self.activenodes], levels=30)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI_CONV, levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[1])

        axs[2].set_xlim(self.Xmin,self.Xmax)
        axs[2].set_ylim(self.Ymin,self.Ymax)
        a = axs[2].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], np.log(error), levels=30)
        axs[2].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        plt.colorbar(a, ax=axs[2])

        plt.show()
        
        return