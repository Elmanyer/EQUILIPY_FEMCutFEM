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


# This script contains the Python object defining a plasma equilibrium problem, 
# modeled using the Grad-Shafranov PDE for an axisymmetrical system such as a tokamak. 
 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import os
import shutil
from random import random
from scipy.interpolate import griddata
from src.GaussQuadrature import *
from src.ShapeFunctions import *
from src.Element import *
from src.Magnet import *

class GradShafranovCutFEM:
    
    # GENERAL PARAMETERS
    epsilon0 = 8.8542E-12        # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19                # J eV-1   Botlzmann constant

    def __init__(self,MESH,CASE):
        # WORKING DIRECTORY
        pwd = os.getcwd()
        self.pwd = pwd[:pwd.rfind("EQUILIPY")+9]
        
        # INPUT FILES:
        self.mesh_folder = self.pwd + '/MESHES/' + MESH
        self.MESH = MESH[MESH.rfind("TS-CUTFEM-")+10:]
        self.case_file = self.pwd + '/CASES/' + CASE
        self.CASE = CASE[CASE.rfind('/')+1:]
        
        # OUTPUT FILES
        self.outputdir = self.pwd + '/RESULTS/' + self.CASE + '-' + self.MESH
        self.PARAMS_file = None             # OUTPUT FILE CONTAINING THE SIMULATION PARAMETERS 
        self.PSI_file = None                # OUTPUT FILE CONTAINING THE PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.PSIcrit_file = None            # OUTPUT FILE CONTAINING THE CRITICAL PSI VALUES
        self.PSI_NORM_file = None           # OUTPUT FILE CONTAINING THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.PSI_B_file = None              # OUTPUT FILE CONTAINING THE PSI_B BOUNDARY VALUES
        self.RESIDU_file = None             # OUTPUT FILE CONTAINING THE RESIDU FOR EACH ITERATION
        self.ElementsClassi_file = None     # OUTPUT FILE CONTAINING THE CLASSIFICATION OF MESH ELEMENTS
        self.PlasmaLevSetVals_file = None   # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.VacVessLevSetVals_file = None  # OUTPUT FILE CONTAINING THE VACUUM VESSEL BOUNDARY LEVEL-SET FIELD VALUES
        self.L2error_file = None            # OUTPUT FILE CONTAINING THE ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        self.ELMAT_file = None              # OUTPUT FILE CONTAINING THE ELEMENTAL MATRICES FOR EACH ITERATION
        
        # OUTPUT SWITCHS
        self.PARAMS_output = False            # OUTPUT SWITCH FOR SIMULATION PARAMETERS 
        self.PSI_output = False               # OUTPUT SWITCH FOR PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.PSIcrit_output = False           # OUTPUT SWITCH FOR CRITICAL PSI VALUES
        self.PSI_NORM_output = False          # OUTPUT SWITCH FOR THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.PSI_B_output = False             # OUTPUT SWITCH FOR PSI_B BOUNDARY VALUES
        self.RESIDU_output = False            # OUTPUT SWITCH FOR RESIDU FOR EACH ITERATION
        self.ElementsClassi_output = False    # OUTPUT SWITCH FOR CLASSIFICATION OF MESH ELEMENTS
        self.PlasmaLevSetVals_output = False  # OUTPUT SWITCH FOR PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.VacVessLevSetVals_output = False # OUTPUT SWITCH FOR VACUUM VESSEL BOUNDARY LEVEL-SET FIELD VALUES
        self.L2error_output = False           # OUTPUT SWITCH FOR ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        self.ELMAT_output = False             # OUTPUT SWITCH FOR ELEMENTAL MATRICES
        self.plotElemsClassi_output = False   # OUTPUT SWITCH FOR ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
        self.plotPSI_output = False           # OUTPUT SWITCH FOR PSI SOLUTION PLOTS AT EACH ITERATION
        
        # PROBLEM CASE PARAMETERS
        self.PLASMA_BOUNDARY = None         # PLASMA BOUNDARY BEHAVIOUR: self.FIXED_BOUNDARY  or  self.FREE_BOUNDARY
        self.PLASMA_GEOMETRY = None         # PLASMA REGION GEOMETRY: self.FIRST_WALL or self.F4E_BOUNDARY 
        self.PLASMA_CURRENT = None          # PLASMA CURRENT MODELISATION: self.LINEAR_CURRENT, self.NONLINEAR_CURRENT or self.PROFILES_CURRENT
        self.VACUUM_VESSEL = None           # VACUUM VESSEL GEOMETRY: self.COMPUTATIONAL or self.FIRST_WALL
        self.TOTAL_CURRENT = None           # TOTAL CURRENT IN PLASMA
        
        # ELEMENTAL CLASSIFICATION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = None        # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.VacVessWallElems = None        # LIST OF CUT (OR NOT) ELEMENT'S INDEXES, CONTAINING VACUUM VESSEL FIRST WALL (OR COMPUTATIONAL DOMAIN'S BOUNDARY)
        self.ExteriorElems = None           # LIST OF CUT ELEMENT'S INDEXES LYING ON THE VACUUM VESSEL FIRST WALL EXTERIOR REGION
        self.NonCutElems = None             # LIST OF ALL NON CUT ELEMENTS
        self.CutElems = None                # LIST OF ALL CUT ELEMENTS
        self.Elements = None                # ARRAY CONTAINING ALL ELEMENTS IN MESH (PYTHON OBJECTS)
        
        # ARRAYS
        self.PlasmaBoundLevSet = None       # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.VacVessWallLevSet = None       # VACUUM VESSEL FIRST WALL GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.PSI = None                     # PSI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PSI POINTS
        self.PSI_0 = None                   # PSI VALUE AT MAGNETIC AXIS MINIMA
        self.PSI_X = None                   # PSI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PSI_NORM = None                # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_B = None                   # VACUUM VESSEL WALL PSI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_CONV = None                # CONVERGED NORMALISED PSI SOLUTION FIELD 
        self.residu_INT = None              # INTERNAL LOOP RESIDU
        self.residu_EXT = None              # EXTERNAL LOOP RESIDU
        
        # VACCUM VESSEL FIRST WALL GEOMETRY
        self.epsilon = None                 # VACUUM VESSEL INVERSE ASPECT RATIO
        self.kappa = None                   # VACUUM VESSEL ELONGATION
        self.delta = None                   # VACUUM VESSEL TRIANGULARITY
        self.R0 = None                      # VACUUM VESSEL MEAN RADIUS

        # PARAMETRISED INITIAL PLASMA EQUILIBRIUM GUESS
        self.CONTROL_POINTS = None          # NUMBER OF CONTROL POINTS
        self.R_SADDLE = None                # R COORDINATE OF ACTIVE SADDLE POINT
        self.Z_SADDLE = None                # Z COORDINATE OF ACTIVE SADDLE POINT
        self.R_RIGHTMOST = None             # R COORDINATE OF POINT ON THE RIGHT
        self.Z_RIGHTMOST = None             # Z COORDINATE OF POINT ON THE RIGHT
        self.R_LEFTMOST = None              # R COORDINATE OF POINT ON THE LEFT
        self.Z_LEFTMOST = None              # Z COORDINATE OF POINT ON THE LEFT
        self.R_TOP = None                   # R COORDINATE OF POINT ON TOP
        self.Z_TOP = None                   # Z COORDINATE OF POINT ON TOP
                
        ###### FOR FREE-BOUNDARY PROBLEM
        # CONFINING MAGNETS
        self.Ncoils = None                  # TOTAL NUMBER OF COILS
        self.COILS = None                   # ARRAY OF COIL OBJECTS
        self.Nsolenoids = None              # TOTAL NUMBER OF SOLENOIDS
        self.SOLENOIDS = None               # ARRAY OF SOLENOID OBJECTS
        
        # PRESSURE AND TOROIDAL FIELD PROFILES
        self.B0 = None                      # TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
        self.q0 = None                      # TOKAMAK SAFETY FACTOR
        self.P0 = None                      # PRESSURE PROFILE FACTOR
        self.n_p = None                     # EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION
        self.G0 = None                      # TOROIDAL FIELD FACTOR
        self.n_g = None                     # EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
        
        ########################
        # COMPUTATIONAL MESH
        self.ElTypeALYA = None              # TYPE OF ELEMENTS CONSTITUTING THE MESH, USING ALYA NOTATION
        self.ElType = None                  # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = None                 # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC,   3: CUBIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.numedges = None                # NUMBER OF EDGES PER ELEMENT (= 3 IF TRIANGULAR; = 4 IF QUADRILATERAL)
        self.nedge = None                   # NUMBER OF NODES ON ELEMENTAL EDGE
        self.dim = None                     # SPACE DIMENSION
        self.Tbound = None                  # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        self.Nbound = None                  # NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
        self.Nnbound = None                 # NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodes = None           # LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.Xmax = None                    # COMPUTATIONAL MESH MAXIMAL X (R) COORDINATE
        self.Xmin = None                    # COMPUTATIONAL MESH MINIMAL X (R) COORDINATE
        self.Ymax = None                    # COMPUTATIONAL MESH MAXIMAL Y (Z) COORDINATE
        self.Ymin = None                    # COMPUTATIONAL MESH MINIMAL Y (Z) COORDINATE
        self.NnFW = None                    # NUMBER OF NODES ON VACUUM VESSEL FIRST WALL APPROXIMATION
        self.XFW = None                     # COORDINATES MATRIX FOR NODES ON VACCUM VESSEL FIRST WALL APPROXIMATION
        self.NnPB = None                    # NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        
        # NUMERICAL TREATMENT PARAMETERS
        self.QuadratureOrder = None         # NUMERICAL INTEGRATION QUADRATURE ORDER
        self.SoleOrder = 2                  # SOLENOID ELEMENT (BAR) ORDER
        
        #### DOBLE WHILE LOOP STRUCTURE PARAMETERS
        self.INT_TOL = None                 # INTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.EXT_TOL = None                 # EXTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.INT_ITER = None                # INTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.EXT_ITER = None                # EXTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.converg_EXT = None             # EXTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.converg_INT = None             # INTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.it_EXT = None                  # EXTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it_INT = None                  # INTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it = 0                         # TOTAL NUMBER OF ITERATIONS COUNTER
        self.alpha = None                   # AIKTEN'S SCHEME RELAXATION CONSTANT
        self.gamma = None                   # PLASMA TOTAL CURRENT CORRECTION FACTOR
        #### BOUNDARY CONSTRAINTS
        self.beta = None                    # NITSCHE'S METHOD PENALTY TERM
        #### OPTIMIZATION OF CRITICAL POINTS
        self.EXTR_R0 = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS R COORDINATE
        self.EXTR_Z0 = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS Z COORDINATE
        self.SADD_R0 = None                 # SADDLE POINT OPTIMIZATION INITIAL GUESS R COORDINATE
        self.SADD_Z0 = None                 # SADDLE POINT OPTIMIZATION INITIAL GUESS Z COORDINATE
        
        # EQUILIPY PARAMETRISATION WITH FLAGS
        #### PLASMA BOUNDARY BEHAVIOUR FLAGS
        self.FIXED_BOUNDARY = 0
        self.FREE_BOUNDARY = 1
        #### GEOMETRY FLAGS
        self.F4E_BOUNDARY = 0
        self.FIRST_WALL = 1
        self.COMPUTATIONAL = 2
        #### PLASMA MODEL PARAMETERS
        self.LINEAR_CURRENT = 0
        self.NONLINEAR_CURRENT = 1
        self.ZHENG_CURRENT = 2
        self.PROFILES_CURRENT = 3
        #### INTERFACE FLAGS
        self.PLASMAbound = 0
        self.VACVESbound = 1
        
        # PLASMA MODELS COEFFICIENTS
        self.coeffsLINEAR = None                       # TOKAMAK FIRST WALL LEVEL-0 CONTOUR COEFFICIENTS (LINEAR PLASMA MODEL CASE SOLUTION)
        self.coeffsNONLINEAR = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0] 
        self.coeffsZHENG = None
        
        self.activenodes = None
        self.L2error = None
        
        return
    
    def print_all_attributes(self):
        """ Function which prints all object EQUILI attributes and their corresponding values. """
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")
        return
    
    def ALYA2Py(self):
        match self.ElTypeALYA:
            case 2:
                self.ElType = 0
                self.ElOrder = 1
            case 3:
                self.ElType = 0
                self.ElOrder = 2
            case 4:
                self.ElType = 0
                self.ElOrder = 3
            case 10:
                self.ElType = 1
                self.ElOrder = 1
            case 11:
                self.ElType = 1
                self.ElOrder = 2
            case 16:
                self.ElType = 1
                self.ElOrder = 3
            case 12:
                self.ElType = 2
                self.ElOrder = 1
            case 14:
                self.ElType = 2
                self.ElOrder = 2
            case 15:
                self.ElType = 2
                self.ElOrder = 3
        return
    
    
    ##################################################################################################
    ############################### READ INPUT DATA FILES ############################################
    ##################################################################################################
    
    def ReadMesh(self):
        """ Reads input mesh data files. """
        
        print("     -> READ MESH DATA FILES...",end='')
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
            elif l[0] == '  TYPES_OF_ELEMENTS':
                self.ElTypeALYA = int(l[1])
            elif l[0] == '  BOUNDARIES':  # read number of boundaries
                self.Nbound = int(l[1])
        file.close()
        
        self.ALYA2Py()
        
        # NUMBER OF NODES PER ELEMENT
        self.n, self.nedge = ElementalNumberOfNodes(self.ElType, self.ElOrder)
        if self.ElType == 1:
            self.numedges = 3
        elif self.ElType == 2:
            self.numedges = 4
        
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
                    self.Tbound[k,m] = int(l[m+1])
                k += 1
        file.close()
        # PYTHON INDEXES START AT 0 AND NOT AT 1. THUS, THE CONNECTIVITY MATRIX INDEXES MUST BE ADAPTED
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
            if line[0] == 'PLASB:':          # READ PLASMA BOUNDARY CONDITION (FIXED OR FREE)
                if line[1] == 'FIXED':
                    self.PLASMA_BOUNDARY =  self.FIXED_BOUNDARY
                elif line[1] == 'FREED':
                    self.PLASMA_BOUNDARY = self.FREE_BOUNDARY
            elif line[0] == 'PLASG:':        # READ PLASMA REGION GEOMETRY (VACUUM VESSEL FIRST WALL OR F4E TRUE SHAPE)
                if line[1] == 'FIRST':
                    self.PLASMA_GEOMETRY = self.FIRST_WALL
                elif line[1] == 'COMPU':
                    self.PLASMA_GEOMETRY = self.COMPUTATIONAL
                elif line[1] == 'PARAM':
                    self.PLASMA_GEOMETRY = self.F4E_BOUNDARY
            elif line[0] == 'PLASC:':         # READ MODEL FOR PLASMA CURRENT (LINEAR, NONLINEAR, ZHENG OR DEFINED USING PROFILES FOR PRESSURE AND TOROIDAL FIELD)
                if line[1] == 'LINEA':
                    self.PLASMA_CURRENT = self.LINEAR_CURRENT
                elif line[1] == 'NONLI':
                    self.PLASMA_CURRENT = self.NONLINEAR_CURRENT
                elif line[1] == 'ZHENG':
                    self.PLASMA_CURRENT = self.ZHENG_CURRENT
                elif line[1] == 'PROFI':
                    self.PLASMA_CURRENT = self.PROFILES_CURRENT
                else:
                    self.PLASMA_CURRENT = l[1]
            elif line[0] == 'VACVE:':          # READ VACUUM VESSEL GEOMETRY (RECTANGLE -> COMPUTATIONAL DOMAIN BOUNDARY ; FIRST_WALL -> USE FIRST WALL GEOMETRY)
                if line[1] == 'FIRST':
                    self.VACUUM_VESSEL = self.FIRST_WALL
                elif line[1] == 'COMPU':
                    self.VACUUM_VESSEL = self.COMPUTATIONAL
                elif line[1] == 'PARAM':
                    self.VACUUM_VESSEL = self.F4E_BOUNDARY
            elif line[0] == 'TOTAL_CURRENT:':        # READ TOTAL PLASMA CURRENT
                self.TOTAL_CURRENT = float(line[1])
            return
        
        def BlockFirstWall(self,line):
            if line[0] == 'R0TOK:':          # READ TOKAMAK FIRST WALL MAJOR RADIUS 
                self.R0 = float(line[1])
            elif line[0] == 'EPSILON:':      # READ TOKAMAK FIRST WALL INVERSE ASPECT RATIO
                self.epsilon = float(line[1])
            elif line[0] == 'KAPPA:':        # READ TOKAMAK FIRST WALL ELONGATION 
                self.kappa = float(line[1])
            elif line[0] == 'DELTA:':        # READ TOKAMAK FIRST WALL TRIANGULARITY 
                self.delta = float(line[1])
            return
        
        def BlockF4E(self,line):
            # READ PLASMA SHAPE CONTROL POINTS
            if line[0] == 'CONTROL_POINTS:':    # READ INITIAL PLASMA REGION NUMBER OF CONTROL POINTS
                self.CONTROL_POINTS = int(line[1])
            elif line[0] == 'R_SADDLE:':    # READ INITIAL PLASMA REGION SADDLE POINT R COORDINATE
                self.R_SADDLE = float(line[1])
            elif line[0] == 'Z_SADDLE:':    # READ INITIAL PLASMA REGION SADDLE POINT Z COORDINATE
                self.Z_SADDLE = float(line[1])
            elif line[0] == 'R_RIGHTMOST:':    # READ INITIAL PLASMA REGION RIGHT POINT R COORDINATE
                self.R_RIGHTMOST = float(line[1])
            elif line[0] == 'Z_RIGHTMOST:':    # READ INITIAL PLASMA REGION RIGHT POINT Z COORDINATE 
                self.Z_RIGHTMOST = float(line[1])
            elif line[0] == 'R_LEFTMOST:':    # READ INITIAL PLASMA REGION LEFT POINT R COORDINATE 
                self.R_LEFTMOST = float(line[1])
            elif line[0] == 'Z_LEFTMOST:':    # READ INITIAL PLASMA REGION LEFT POINT Z COORDINATE 
                self.Z_LEFTMOST = float(line[1])
            elif line[0] == 'R_TOPP:':    # READ INITIAL PLASMA REGION TOP POINT R COORDINATE 
                self.R_TOP = float(line[1])
            elif line[0] == 'Z_TOPP:':    # READ INITIAL PLASMA REGION TOP POINT Z COORDINATE 
                self.Z_TOP = float(line[1])
            return
        
        def BlockExternalMagnets(self,line,i,j):
            if line[0] == 'N_COILS:':    # READ TOTAL NUMBER COILS 
                self.Ncoils = int(line[1])
                self.COILS = [Coil(index = icoil, dim=self.dim, X=np.zeros([self.dim]), I=None) for icoil in range(self.Ncoils)] 
            elif line[0] == 'Rposi:' and i<self.Ncoils:    # READ i-th COIL X POSITION
                self.COILS[i].X[0] = float(line[1])
            elif line[0] == 'Zposi:' and i<self.Ncoils:    # READ i-th COIL Y POSITION
                self.COILS[i].X[1] = float(line[1])
            elif line[0] == 'Inten:' and i<self.Ncoils:    # READ i-th COIL INTENSITY
                self.COILS[i].I = float(line[1])
                i += 1
            # READ SOLENOID PARAMETERS:
            elif line[0] == 'N_SOLENOIDS:':    # READ TOTAL NUMBER OF SOLENOIDS
                self.Nsolenoids = int(line[1])
                self.SOLENOIDS = [Solenoid(index = isole, ElOrder=self.ElOrder, dim=self.dim, Xe=np.zeros([2,self.dim]), I=None) for isole in range(self.Nsolenoids)] 
            elif line[0] == 'Rposi:' and j<self.Nsolenoids:    # READ j-th SOLENOID X POSITION
                self.SOLENOIDS[j].Xe[0,0] = float(line[1])
                self.SOLENOIDS[j].Xe[1,0] = float(line[1])
            elif line[0] == 'Zlowe:' and j<self.Nsolenoids:     # READ j-th SOLENOID Y POSITION
                self.SOLENOIDS[j].Xe[0,1] = float(line[1])
            elif line[0] == 'Zuppe:' and j<self.Nsolenoids:      # READ j-th SOLENOID Y POSITION
                self.SOLENOIDS[j].Xe[1,1] = float(line[1])
                self.SOLENOIDS[j].ComputeHOnodes()
            elif line[0] == 'Inten:' and j<self.Nsolenoids:    # READ j-th SOLENOID INTENSITY
                self.SOLENOIDS[j].I = float(line[1])
                j += 1
            return i, j
        
        def BlockProfiles(self,line):
            if line[0] == 'B0_equ:':    # READ TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
                self.B0 = float(line[1])
            elif line[0] == 'q0_equ:':    # READ TOKAMAK SAFETY FACTOR 
                self.q0 = float(line[1])
            elif line[0] == 'np_equ:':    # READ EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION 
                self.n_p = float(line[1])
            elif line[0] == 'g0_equ:':    # READ TOROIDAL FIELD PROFILE FACTOR
                self.G0 = float(line[1])
            elif line[0] == 'ng_equ:':    # READ EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
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
            elif line[0] == 'BETA_equ:':             # READ NITSCHE'S METHOD PENALTY PARAMETER 
                self.beta = float(line[1])
            elif line[0] == 'RELAXATION:':       # READ AITKEN'S METHOD RELAXATION PARAMETER
                self.alpha = float(line[1])
            elif line[0] == 'EXTR_R0:':	        # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
                self.EXTR_R0 = float(line[1])
            elif line[0] == 'EXTR_Z0:':           # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
                self.EXTR_Z0 = float(line[1])
            elif line[0] == 'SADD_R0:':           # ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
                self.SADD_R0 = float(line[1])
            elif line[0] == 'SADD_Z0:': 
                self.SADD_Z0 = float(line[1])
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
                if self.PLASMA_GEOMETRY == self.F4E_BOUNDARY:
                    BlockF4E(self,l)
                # READ PARAMETERS FOR PRESSURE AND TOROIDAL FIELD PROFILES
                if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
                    BlockProfiles(self,l)
                # READ COIL PARAMETERS
                if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                    i,j = BlockExternalMagnets(self,l,i,j)
                # READ NUMERICAL TREATMENT PARAMETERS
                BlockNumericalTreatement(self,l)
                
        if self.PLASMA_GEOMETRY == self.VACUUM_VESSEL:
            raise Exception("PLASMA REGION GEOMETRY AND VACUUM VESSEL FIRST WALL GEOMETRY MUST BE DIFFERENT")
        
        # TOKAMAK'S 1rst WALL GEOMETRY COEFFICIENTS, USED ALSO FOR LINEAR PLASMA MODEL ANALYTICAL SOLUTION (INITIAL GUESS)
        self.coeffsLINEAR = self.ComputeLinearSolutionCoefficients()
        
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            # COMPUTE PRESSURE PROFILE FACTOR
            self.P0=self.B0*((self.kappa**2)+1)/(self.mu0*(self.R0**2)*self.q0*self.kappa)
        
        # LOOK FOR EXTREMUM IN LINEAR OR NONLINEAR ANALITYCAL SOLUTIONS IN ORDER TO NORMALISE THE CONSTRAINED VALUES ON THE INTERFACE FOR FIXED BOUNDARY PROBLEM
        self.PSIextr_analytical = 1.0
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY and self.PLASMA_CURRENT in [self.LINEAR_CURRENT, self.NONLINEAR_CURRENT]:
            X0 = np.array([self.R0,0])
            match self.PLASMA_CURRENT:
                case self.LINEAR_CURRENT:
                    self.Xcrit_analytical = optimize.minimize(self.PSIAnalyticalSolution, X0, args=(self.PLASMA_CURRENT)).x
                    self.PSIextr_analytical = self.PSIAnalyticalSolution(self.Xcrit_analytical,self.PLASMA_CURRENT)
                case self.NONLINEAR_CURRENT:
                    def minusPSIanal(X):
                        return -self.PSIAnalyticalSolution(X,self.NONLINEAR_CURRENT)
                    self.Xcrit_analytical = optimize.minimize(minusPSIanal,X0).x
                    self.PSIextr_analytical = self.PSIAnalyticalSolution(self.Xcrit_analytical,self.PLASMA_CURRENT)
        
        print('Done!')  
        return
    
    def ReadFixdata(self):
        print("     -> READ FIX DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        FixDataFile = self.mesh_folder +'/'+ 'TS-CUTFEM-' + self.MESH +'.fix.dat'
        file = open(FixDataFile, 'r') 
        self.BoundaryIden = np.zeros([self.Nbound],dtype=int)
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            
            if l[0] == "ON_BOUNDARIES" or l[0] == "END_ON_BOUNDARIES":
                pass
            else:
                self.BoundaryIden[int(l[0])-1] = int(l[1])
                
        # DEFINE THE DIFFERENT SETS OB BOUNDARY NODES
        self.BoundaryNodesSets = [set(),set()]
        for iboun in range(self.Nbound):
            for node in self.Tbound[iboun,:-1]:
                self.BoundaryNodesSets[self.BoundaryIden[iboun]-1].add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodesSets[0] = np.array(sorted(self.BoundaryNodesSets[0]))
        self.BoundaryNodesSets[1] = np.array(sorted(self.BoundaryNodesSets[1]))
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
        
        self.coeffsZHENG = np.zeros([6])
        
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
            case self.LINEAR_CURRENT:
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0
                # ANALYTICAL SOLUTION
                PSIexact = (Xstar[0]**4)/8 + self.coeffsLINEAR[0] + self.coeffsLINEAR[1]*Xstar[0]**2 + self.coeffsLINEAR[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
                
            case self.NONLINEAR_CURRENT:
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0 
                # ANALYTICAL SOLUTION
                PSIexact = np.sin(self.coeffsNONLINEAR[0]*(Xstar[0]+self.coeffsNONLINEAR[2]))*np.cos(self.coeffsNONLINEAR[1]*Xstar[1])  
                
            case self.ZHENG_CURRENT:
                # ANALYTICAL SOLUTION
                PSIexact = self.coeffsZHENG[0]+self.coeffsZHENG[1]*X[0]**2+self.coeffsZHENG[2]*(X[0]**4-4*X[0]**2*X[1]**2)+self.coeffsZHENG[3]*(np.log(X[0])
                                    *X[0]**2-X[1]**2)+(self.coeffsZHENG[4]*X[0]**4)/8 - (self.coeffsZHENG[5]*X[1]**2)/2
                
            case "FAKE":
                PSIexact = np.sin(2*X[0])+0.1*X[1]**3
        
        return PSIexact
    
    ##################################################################################################
    ###################################### PLASMA CURRENT ############################################
    ##################################################################################################
    
    def Jphi(self,X,PSI):
        # COMPUTES THE SOURCE TERM Jphi, WHICH GOES IN THE GRAD-SHAFRANOV EQ. RIGHT-HAND-SIDE     mu0*R*Jphi
        match self.PLASMA_CURRENT:
            case self.LINEAR_CURRENT:
                # COMPUTE LINEAR MODEL PLASMA CURRENT
                Jphi = X[0]/self.mu0
            
            case self.NONLINEAR_CURRENT: 
                # COMPUTE NONLINEAR MODEL PLASMA CURRENT
                Kr, Kz, r0 = self.coeffsNONLINEAR
                Jphi = -((Kr**2+Kz**2)*PSI+(Kr/X[0])*np.cos(Kr*(X[0]+r0))*np.cos(Kz*X[1])+X[0]*(np.sin(Kr*(X[0]+r0))**2*np.cos(Kz*X[1])**2
                            -PSI**2+np.exp(-np.sin(Kr*(X[0]+r0))*np.cos(Kz*X[1]))-np.exp(-PSI)))/(X[0]*self.mu0)
            
            case self.ZHENG_CURRENT:
                # COMPUTE PLASMA CURRENT MODEL BASED ON ZHENG PAPER
                Jphi = (self.coeffsZHENG[4]*X[0]**2 - self.coeffsZHENG[5])/ (X[0]*self.mu0)
        
            case self.PROFILES_CURRENT:
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
    
    
    def SourceTerm(self,X,PSI):
        match self.PLASMA_CURRENT:
            case self.LINEAR_CURRENT:
                Xstar = X/self.R0
                # COMPUTE LINEAR MODEL PLASMA CURRENT
                source = Xstar[0]**2
            
            case self.NONLINEAR_CURRENT: 
                Xstar = X/self.R0
                # COMPUTE NONLINEAR MODEL PLASMA CURRENT
                Kr, Kz, r0 = self.coeffsNONLINEAR
                source = -((Kr**2+Kz**2)*PSI+(Kr/Xstar[0])*np.cos(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1])+Xstar[0]*(np.sin(Kr*(Xstar[0]+r0))**2*np.cos(Kz*Xstar[1])**2
                            -PSI**2+np.exp(-np.sin(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1]))-np.exp(-PSI)))
                
            case self.ZHENG_CURRENT:
                # COMPUTE PLASMA CURRENT MODEL BASED ON ZHENG PAPER
                source = self.coeffsZHENG[4]*X[0]**2 - self.coeffsZHENG[5]
                
            case self.PROFILES_CURRENT:
                source = -self.mu0*X[0]**2*self.dPdPSI(PSI) - 0.5*self.dG2dPSI(PSI)
                
        if self.PLASMA_CURRENT == 'FAKE':
            source = -4*np.sin(2*X[0]) + 0.6*X[1]

        return source
        
        
    
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
        FOR self.FREE_BOUNDARY BOUNDARY PROBLEM, SUCH VALUES ARE OBTAINED BY ACCOUNTING FOR THE CONTRIBUTIONS FROM THE EXTERNAL
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
    
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:  
            k = 0
            # COMPUTE PSI_B VALUE...
            for ielem in self.VacVessWallElems:                      # FOR EACH VACUUM VESSEL ELEMENT ...
                for INTERFACE in self.Elements[ielem].InterfApprox:  # INTERFACE APPROXIMATION ...
                    for SEGMENT in INTERFACE.Segments:              # SEGMENT ELEMENT ...
                        for igpoint in range(SEGMENT.ng):    # GAUSS INTEGRATION POINT
                            # ISOLATE NODAL COORDINATES
                            Xnode = SEGMENT.Xg[igpoint,:]
                            
                            # CONTRIBUTION FROM EXTERNAL COILS CURRENT 
                            for COIL in self.COILS: 
                                PSI_B[k] += self.mu0 * GreenFunction(Xnode,COIL.X) * COIL.I
                            
                            # CONTRIBUTION FROM EXTERNAL SOLENOIDS CURRENT  ->>  INTEGRATE OVER SOLENOID LENGTH 
                            for SOLENOID in self.SOLENOIDS:
                                Jsole = SOLENOID.I/np.linalg.norm(SOLENOID.Xe[0,:]-SOLENOID.Xe[1,:])   # SOLENOID CURRENT LINEAR DENSITY
                                # LOOP OVER GAUSS NODES
                                for ig in range(SOLENOID.ng):
                                    for l in range(SOLENOID.n):
                                        PSI_B[k] += self.mu0 * GreenFunction(Xnode,SOLENOID.Xg[ig,:]) * Jsole * SOLENOID.Ng[ig,l] * SOLENOID.detJg[ig] * SOLENOID.Wg[ig]
                                        
                            # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                            #   1. INTEGRATE IN PLASMA ELEMENTS
                            for jelem in self.PlasmaElems:
                                # ISOLATE ELEMENT OBJECT
                                ELEMENT = self.Elements[jelem]
                                # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                                PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                                # LOOP OVER GAUSS NODES
                                for ig in range(ELEMENT.ng):
                                    for l in range(ELEMENT.n):
                                        PSI_B[k] += self.mu0 * GreenFunction(Xnode, ELEMENT.Xg[ig,:])*self.Jphi(ELEMENT.Xg[ig,:],
                                                                    PSIg[ig])*ELEMENT.Ng[ig,l]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]*self.gamma
                                                
                            #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                            for jelem in self.PlasmaBoundElems:
                                # ISOLATE ELEMENT OBJECT
                                ELEMENT = self.Elements[jelem]
                                # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                                for SUBELEM in ELEMENT.SubElements:
                                    if SUBELEM.Dom < 0:  # IN PLASMA REGION
                                        # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                                        PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                                        # LOOP OVER GAUSS NODES
                                        for ig in range(SUBELEM.ng):
                                            for l in range(SUBELEM.n):
                                                PSI_B[k] += self.mu0 * GreenFunction(Xnode, SUBELEM.Xg[ig,:])*self.Jphi(SUBELEM.Xg[ig,:],
                                                                    PSIg[ig])*SUBELEM.Ng[ig,l]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]*self.gamma   
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
            for i in range(self.numedges-1):
                # FIND ELEMENTS LYING ON THE INTERFACE (LEVEL-SET VERTICES VALUES EQUAL TO 0 OR WITH DIFFERENT SIGN)
                if LSe[i] == 0:  # if node is on Level-Set 0 contour
                    region = 0
                    break
                elif np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between vertices values change -> INTERFACE ELEMENT
                    region = 0
                    break
                # FIND ELEMENTS LYING INSIDE A SPECIFIC REGION (LEVEL-SET VERTICES VALUES WITH SAME SIGN)
                else:
                    if i+2 == self.numedges:   # if all vertices values have the same sign
                        # LOCATE ON WHICH REGION LIES THE ELEMENT
                        if np.sign(LSe[i+1]) > 0:   # all vertices values with positive sign -> EXTERIOR REGION ELEMENT
                            region = +1
                        else:   # all vertices values with negative sign -> INTERIOR REGION ELEMENT 
                            region = -1
                            
                        # CHECK LEVEL-SET SIGN ON ELEMENTAL 'HIGH ORDER' NODES
                        #for i in range(self.numedges,self.n-self.numedges):  # LOOP OVER NODES WHICH ARE NOT ON VERTICES
                        for i in range(self.numedges,self.n):
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
                self.OLDplasmaBoundLevSet = self.PlasmaBoundLevSet.copy()
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
        if self.VACUUM_VESSEL == self.COMPUTATIONAL:
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
            #X0_extr = np.array([6,0])
            #X0_saddle = np.array([5,-4])
            X0_extr = np.array([self.EXTR_R0,self.EXTR_Z0],dtype=float)
            X0_saddle = np.array([self.SADD_R0,self.SADD_Z0],dtype=float)
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
            if self.it == 1:
                print("LOCAL EXTREMUM NOT FOUND. TAKING SOLUTION AT INITIAL GUESS")
                elem = self.SearchElement(X0_extr,self.PlasmaElems)
                self.Xcrit[1,0,:-1] = X0_extr
                self.Xcrit[1,0,-1] = elem
            else:
                print("LOCAL EXTREMUM NOT FOUND. TAKING PREVIOUS SOLUTION")
                self.Xcrit[1,0,:] = self.Xcrit[0,0,:]
            
        # INTERPOLATE PSI VALUE ON CRITICAL POINT
        self.PSI_0 = self.Elements[int(self.Xcrit[1,0,-1])].ElementalInterpolationPHYSICAL(self.Xcrit[1,0,:-1],PSI[self.Elements[int(self.Xcrit[1,0,-1])].Te]) 
        print('LOCAL EXTREMUM AT ',self.Xcrit[1,0,:-1],' (ELEMENT ', int(self.Xcrit[1,0,-1]),') WITH VALUE PSI_0 = ',self.PSI_0)
            
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
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
            self.PSI_X = self.Elements[int(self.Xcrit[1,1,-1])].ElementalInterpolationPHYSICAL(self.Xcrit[1,1,:-1],PSI[self.Elements[int(self.Xcrit[1,1,-1])].Te]) 
            print('SADDLE POINT AT ',self.Xcrit[1,1,:-1],' (ELEMENT ', int(self.Xcrit[1,1,-1]),') WITH VALUE PSI_X = ',self.PSI_X)
        
        else:
            self.Xcrit[1,1,:-1] = [self.Xmin,self.Ymin]
            self.PSI_X = 0
            
        return 
    
    
    def NormalisePSI(self):
        # NORMALISE SOLUTION OBTAINED FROM SOLVING CutFEM SYSTEM OF EQUATIONS USING CRITICAL PSI VALUES, PSI_0 AND PSI_X
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY or self.PLASMA_CURRENT == self.PROFILES_CURRENT:
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
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
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
    
    def UpdateElementalPSIg(self,BOUNDARY):
        """ FUNCTION WHICH UPDATES THE PSI VALUE CONSTRAINTS PSIgseg ON ALL INTERFACE APPROXIMATION SEGMENTS INTEGRATION POINTS. """
        
        if BOUNDARY == self.PLASMAbound:
            for ielem in self.PlasmaBoundElems:
                for INTERFACE in self.Elements[ielem].InterfApprox:  # FOR EACH PLASMA/VACUUM INTERFACE EDGE
                    for SEGMENT in INTERFACE.Segments:
                        # COMPUTE INTERFACE CONDITIONS PSI_D
                        SEGMENT.PSIgseg = np.zeros([SEGMENT.ng])
                        # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE EDGE
                        for ignode in range(SEGMENT.ng):
                            if self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:
                                SEGMENT.PSIgseg[ignode] = self.PSIAnalyticalSolution(SEGMENT.Xg[ignode,:],self.PLASMA_CURRENT)
                            elif self.PLASMA_CURRENT == self.PROFILES_CURRENT:
                                SEGMENT.PSIgseg[ignode] = self.PSI_X
                            else:
                                #EDGE.PSIgseg[point] = 0
                                SEGMENT.PSIgseg[ignode] = self.PSIAnalyticalSolution(SEGMENT.Xg[ignode,:],self.PLASMA_CURRENT)
                
        elif BOUNDARY == self.VACVESbound:
            # FOR FIXED PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PSI_B = 0, THUS SO ARE ALL ELEMENTAL VALUES  ->> PSI_Be = 0
            if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:  
                for ielem in self.VacVessWallElems:
                    for INTERFACE in self.Elements[ielem].InterfApprox:
                        for SEGMENT in INTERFACE.Segments:
                            # COMPUTE INTERFACE CONDITIONS PSI_D
                            SEGMENT.PSIgseg = np.zeros([SEGMENT.ng])
            
            # FOR FREE PLASMA BOUNDARY, THE VACUUM VESSEL BOUNDARY PSI_B VALUES ARE COMPUTED FROM THE GRAD-SHAFRANOV OPERATOR'S GREEN FUNCTION
            # IN ROUTINE COMPUTEPSI_B, HERE WE NEED TO SEND TO EACH BOUNDARY ELEMENT THE PSI_B VALUES OF THEIR CORRESPONDING BOUNDARY NODES
            elif self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:  
                k = 0
                # FOR EACH ELEMENT CONTAINING THE VACUUM VESSEL FIRST WALL
                for ielem in self.VacVessWallElems:
                    # FOR EACH FIRST WALL EDGE
                    for INTERFACE in self.Elements[ielem].InterfApprox:
                        for SEGMENT in INTERFACE.Segments:
                            # FOR EACH INTEGRATION POINT ON THE FIRST WALL EDGE
                            SEGMENT.PSIgseg = np.zeros([SEGMENT.ng])
                            for inode in range(SEGMENT.ng):
                                SEGMENT.PSIgseg[inode] = self.PSI_B[k,0]
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
        for ielem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[ielem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER ELEMENTAL NODES
            for i in range(ELEMENT.n):
                 # LOOP OVER GAUSS NODES
                for ig in range(ELEMENT.ng):
                    integral += fun(ELEMENT.Xg[ig,:],PSIg[ig])*ELEMENT.Ng[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for ielem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[ielem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            integral += fun(SUBELEM.Xg[ig,:],PSIg[ig])*SUBELEM.Ng[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]            
        return integral
    
    def ComputeActiveNodes(self):
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:
            plasma_elems = np.concatenate((self.PlasmaBoundElems,self.PlasmaElems), axis=0)
            self.activenodes = set()
            for ielem in plasma_elems:
                for node in self.T[ielem,:]:
                    self.activenodes.add(node)
            self.activenodes = np.array(list(self.activenodes))
        else:
            self.activenodes = range(self.Nn)
        return
    
    def ComputePlasmaNodes(self):
        self.plasmanodes = set()
        for ielem in self.PlasmaElems:
            for node in self.T[ielem,:]:
                self.plasmanodes .add(node)
        self.plasmanodes = np.array(list(self.plasmanodes))
        
        return
    
    def ComputeInterfaceNumberNodes(self,BOUNDARY):
        if BOUNDARY == self.PLASMAbound:
            elements = self.PlasmaBoundElems
        elif BOUNDARY == self.VACVESbound:
            elements = self.VacVessWallElems 
        nnodes = 0
        for ielem in elements:
            for INTERFACE in self.Elements[ielem].InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    nnodes += SEGMENT.ng
        return nnodes
        
    
    
    ##################### INITIALISATION 
    
    def InitialGuess(self):
        """ This function computes the problem's initial guess, which is taken as the LINEAR CASE SOLUTION WITH SOME RANDOM NOISE. """
        PSI0 = np.zeros([self.Nn])
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT: 
            for i in range(self.Nn):
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.LINEAR_CURRENT)*(-0.5)
        else:     
            for i in range(self.Nn):
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.PLASMA_CURRENT)*2*random()
                #PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.PLASMA_CURRENT)
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
        if self.PLASMA_GEOMETRY == self.FIRST_WALL:  # IN THIS CASE, THE PLASMA REGION SHAPE IS TAKEN AS THE SHAPE OF THE TOKAMAK'S FIRST WALL
            if self.PLASMA_CURRENT == self.ZHENG_CURRENT:
                for i in range(self.Nn):
                    self.PlasmaBoundLevSet[i] = -self.PSIAnalyticalSolution(self.X[i,:],self.ZHENG_CURRENT)
            else: 
                for i in range(self.Nn):
                    self.PlasmaBoundLevSet[i] = self.PSIAnalyticalSolution(self.X[i,:],self.LINEAR_CURRENT)
                
        elif self.PLASMA_GEOMETRY == self.F4E_BOUNDARY:    # IN THIS CASE, THE PLASMA REGION SHAPE IS DESCRIBED USING THE F4E SHAPE CONTROL POINTS
            self.PlasmaBoundLevSet = self.F4E_PlasmaBoundLevSet()
            
        self.VacVessWallLevSet = (-1)*np.ones([self.Nn])
        if self.VACUUM_VESSEL == self.COMPUTATIONAL:
            for node_index in self.BoundaryNodes:
                self.VacVessWallLevSet[node_index] = 0
        elif self.VACUUM_VESSEL == self.FIRST_WALL:
            for i in range(self.Nn):
                self.VacVessWallLevSet[i] = self.PSIAnalyticalSolution(self.X[i,:],self.LINEAR_CURRENT)
            
        return 
    
    def InitialiseElements(self):
        """ Function initialising attribute ELEMENTS which is a list of all elements in the mesh. """
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.PlasmaBoundLevSet[self.T[e,:]],
                                 self.VacVessWallLevSet[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    def InitialisePSI(self):  
        """ INITIALISE PSI VECTORS WHERE THE DIFFERENT SOLUTIONS WILL BE STORED ITERATIVELY DURING THE SIMULATION AND COMPUTE INITIAL GUESS."""
        ####### COMPUTE NUMBER OF NODES ON BOUNDARIES' APPROXIMATIONS
        # COMPUTE NUMBER OF NODES ON VACUUM VESSEL FIRST WALL APPROXIMATION 
        self.NnFW = self.ComputeInterfaceNumberNodes(self.VACVESbound)
        # COMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION 
        self.NnPB = self.ComputeInterfaceNumberNodes(self.PLASMAbound)
        
        ####### INITIALISE PSI VECTORS
        print('         -> INITIALISE PSI ARRAYS...', end="")
        self.PSI = np.zeros([self.Nn])            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORM = np.zeros([self.Nn,2])     # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_B = np.zeros([self.NnFW,2])      # VACUUM VESSEL FIRST WALL PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)    
        self.PSI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION FIELD
        
        ####### COMPUTE INITIAL GUESS AND STORE IT IN ARRAY FOR N=0
        # COMPUTE INITIAL GUESS
        print('         -> COMPUTE INITIAL GUESS FOR PSI_NORM...', end="")
        self.PSI_NORM[:,0] = self.InitialGuess()  
        # ASSIGN VALUES TO EACH ELEMENT
        self.UpdateElementalPSI()
        print('Done!')   
        
        ####### COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PSI_B AND STORE THEM IN ARRAY FOR N=0
        print('         -> COMPUTE INITIAL VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        self.ComputeTotalPlasmaCurrentNormalization()
        self.PSI_B[:,0] = self.ComputePSI_B()
        
        ####### ASSIGN CONSTRAINT VALUES ON PLASMA AND VACCUM VESSEL BOUNDARIES
        print('         -> ASSIGN INITIAL BOUNDARY VALUES...', end="")
        # ASSIGN PLASMA BOUNDARY VALUES
        self.PSI_X = 0   # INITIAL CONSTRAINT VALUE ON SEPARATRIX
        self.UpdateElementalPSIg(self.PLASMAbound)
        # ASSIGN VACUUM VESSEL BOUNDARY VALUES
        self.UpdateElementalPSIg(self.VACVESbound)
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
        self.ComputePlasmaBoundaryApproximation()
        print("Done!")
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        # INITIALISE PSI UNKNOWNS
        print("     -> COMPUTE INITIAL GUESS...")
        self.InitialisePSI()
        self.writePSI()
        self.writePSI_NORM()
        self.writePSI_B()
        print('     Done!')
        return  
    
    ##################### OPERATIONS ON COMPUTATIONAL DOMAIN'S BOUNDARY EDGES #########################
    
    def ComputeFirstWallApproximation(self):
        """ APPROXIMATE/IDENTIFY LINEAR EDGES CONFORMING THE VACUUM VESSEL FIRST WALL GEOMETRY ON EACH EDGE. COMPUTE NORMAL VECTORS FOR EACH EDGE. """
        if self.VACUUM_VESSEL == self.COMPUTATIONAL:
            for elem in self.VacVessWallElems:
                # IDENTIFY COMPUTATIONAL DOMAIN'S BOUNDARIES CONFORMING VACUUM VESSEL FIRST WALL
                self.Elements[elem].ComputationalDomainBoundaryEdges(self.Tbound)  
                # COMPUTE OUTWARDS NORMAL VECTOR
                self.Elements[elem].ComputationalDomainBoundaryNormal(self.Xmax,self.Xmin,self.Ymax,self.Ymin)
        else:
            for inter, elem in enumerate(self.VacVessWallElems):
                # APPROXIMATE VACUUM VESSEL FIRST WALL GEOMETRY CUTTING ELEMENT 
                self.Elements[elem].InterfaceApproximation(inter)  
                # COMPUTE OUTWARDS NORMAL VECTOR
                self.Elements[elem].InterfaceNormal()
        # CHECK NORMAL VECTORS ORTHOGONALITY RESPECT TO INTERFACE EDGES
        self.CheckInterfaceNormalVectors(self.VACVESbound)  
        return
    
    def ComputePlasmaBoundaryApproximation(self):
        """ Compute the coordinates for the points describing the interface linear approximation. """
        for inter, elem in enumerate(self.PlasmaBoundElems):
            # APPROXIMATE PLASMA/VACUUM INTERACE GEOMETRY CUTTING ELEMENT 
            self.Elements[elem].InterfaceApproximation(inter)
            # COMPUTE OUTWARDS NORMAL VECTOR
            self.Elements[elem].InterfaceNormal()
        self.CheckInterfaceNormalVectors(self.PLASMAbound)
        return
    
    def CheckInterfaceNormalVectors(self,BOUNDARY):
        if BOUNDARY == self.PLASMAbound:
            elements = self.PlasmaBoundElems
        elif BOUNDARY == self.VACVESbound:
            elements = self.VacVessWallElems
            
        for elem in elements:
            for INTERFACE in self.Elements[elem].InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    dir = np.array([SEGMENT.Xseg[1,0]-SEGMENT.Xseg[0,0], SEGMENT.Xseg[1,1]-SEGMENT.Xseg[0,1]]) 
                    scalarprod = np.dot(dir,SEGMENT.NormalVec)
                    if scalarprod > 1e-10: 
                        raise Exception('Dot product equals',scalarprod, 'for mesh element', elem, ": Normal vector not perpendicular")
        return
    
    ##################### COMPUTE NUMERICAL INTEGRATION QUADRATURES FOR EACH ELEMENT GROUP 
    
    def ComputeIntegrationQuadratures(self):
        """ ROUTINE WHERE THE INITIAL NUMERICAL INTEGRATION QUADRATURES FOR ALL ELEMENTS IN THE MESH ARE PREPARED. """
        
        # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for elem in self.NonCutElems:
            self.Elements[elem].ComputeStandardQuadrature2D(self.QuadratureOrder)
            
        # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for elem in self.CutElems:
            self.Elements[elem].ComputeAdaptedQuadratures(self.QuadratureOrder)
        
        # FOR BOUNDARY ELEMENTS COMPUTE BOUNDARY QUADRATURE ENTITIES TO INTEGRATE OVER BOUNDARY EDGES
        if self.VACUUM_VESSEL == self.COMPUTATIONAL:   
            for elem in self.VacVessWallElems:
                self.Elements[elem].ComputeComputationalDomainBoundaryQuadrature(self.QuadratureOrder)
                
        # FOR FREE-BOUNDARY PROBLEM, COMPUTE QUADRATURES FOR SOLENOIDS
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
            for SOLENOID in self.SOLENOIDS:
                SOLENOID.ComputeIntegrationQuadrature(self.QuadratureOrder)
                
        return
    
    #################### UPDATE EMBEDED METHOD ##############################
    
    def UpdateElements(self):
        """ FUNCTION WHERE THE DIFFERENT METHOD ENTITIES ARE RECOMPUTED ACCORDING TO THE EVOLUTION OF THE LEVEL-SET DEFINING THE PLASMA REGION. 
        THEORETICALY, THE ONLY ELEMENTS AFFECTED AND THUS NEED TO RECOMPUTE THEIR ENTITIES AS THE PLASMA REGION EVOLVES SHOULD BE:
                - PLASMA ELEMENTS
                - PLASMABOUNDARY ELEMENTS
                - VACUUM ELEMENTS. """
                
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
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
                
                # OBTAIN POINTS CONFORMING THE NEW PLASMA DOMAIN BOUNDARY
                fig, ax = plt.subplots(figsize=(6, 8))
                cs = ax.tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0])
                
                paths = list()
                for item in cs.collections:
                    for path in item.get_paths():
                        containsboundary = False
                        for point in path.vertices:
                            # CHECK IF CONTOUR CONTAINS SADDLE POINT (ONE OF ITS POINTS IS CLOSE ENOUGH)
                            if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3:
                                containsboundary = True
                        if containsboundary:
                            paths.append(path)
                            
                if len(paths) > 1:
                    for path in paths:
                        containscompudombound = False
                        for point in path.vertices:
                            # CHECK IF CONTOUR CONTAINS COMPUTATIONAL DOMAIN BOUNDARY POINTS
                            if np.abs(point[0]-self.Xmax) < 0.2 or np.abs(point[0]-self.Xmin) < 0.2 or np.abs(point[1]-self.Ymax) < 0.1 or np.abs(point[1]-self.Ymin) < 0.1:
                                containscompudombound = True
                        if containscompudombound:
                            paths.remove(path)
                            
                    plasmaboundary = list()
                    for point in paths[0].vertices:
                        plasmaboundary.append(point)
                    plasmaboundary = np.array(plasmaboundary)
                    
                else:
                    plasmaboundary = list()
                    oncontour = False
                    firstpass = True
                    secondpass = False
                    counter = 0
                    for path in paths:
                        for point in path.vertices:
                            if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and firstpass:
                                oncontour = True 
                                firstpass = False
                                plasmaboundary.append(point)
                            elif oncontour:
                                plasmaboundary.append(point)
                                counter += 1
                            if counter > 50:
                                secondpass = True
                            if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and secondpass: 
                                oncontour = False 
                                    
                    plasmaboundary.append(plasmaboundary[0])
                    plasmaboundary = np.array(plasmaboundary)
                
                fig.clear()
                plt.close(fig)

                # Create a Path object for the new plasma domain
                polygon_path = mpath.Path(plasmaboundary)
                # Check if the mesh points are inside the new plasma domain
                inside = polygon_path.contains_points(self.X)
                
                # 1. INVERT SIGN DEPENDING ON SOLUTION PLASMA REGION SIGN
                if self.PSI_0 > 0: # WHEN THE OBTAINED SOLUTION IS POSITIVE INSIDE THE PLASMA
                    self.PlasmaBoundLevSet = -self.PSI_NORM[:,1].copy()
                else: # WHEN THE OBTAINED SOLUTION IS NEGATIVE INSIDE THE PLASMA
                    self.PlasmaBoundLevSet = self.PSI_NORM[:,1].copy()

                # 2. DISCARD POINTS OUTSIDE THE PLASMA REGION
                for inode in range(self.Nn):
                    if not inside[inode]:
                        self.PlasmaBoundLevSet[inode] = np.abs(self.PlasmaBoundLevSet[inode])

                ###### UPDATE PLASMA REGION LEVEL-SET ELEMENTAL VALUES     
                for ELEMENT in self.Elements:
                    ELEMENT.PlasmaLSe = self.PlasmaBoundLevSet[self.T[ELEMENT.index,:]]
                    
                self.ClassifyElements()
                
                ###### RECOMPUTE PLASMA/VACUUM INTERFACE LINEAR APPROXIMATION and NORMAL VECTORS
                self.ComputePlasmaBoundaryApproximation()
                
                ###### RECOMPUTE NUMERICAL INTEGRATION QUADRATURES
                for elem in np.concatenate((self.PlasmaElems, self.VacuumElems), axis = 0):
                    self.Elements[elem].ComputeStandardQuadrature2D(self.QuadratureOrder)
                # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
                for elem in self.PlasmaBoundElems:
                    self.Elements[elem].ComputeAdaptedQuadratures(self.QuadratureOrder)
                    
                # RECOMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION 
                self.NnPB = self.ComputeInterfaceNumberNodes(self.PLASMAbound)
                    
                return
            
    #################### L2 ERROR COMPUTATION ############################
    
    def ComputeL2error(self):
        L2error = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                # LOOP OVER ELEMENTAL NODES
                for i in range(ELEMENT.n):
                    L2error += (PSIg[ig]-self.PSIAnalyticalSolution(ELEMENT.Xg[ig,:],self.PLASMA_CURRENT))**2*ELEMENT.Ng[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            L2error += (PSIg[ig]-self.PSIAnalyticalSolution(SUBELEM.Xg[ig,:],self.PLASMA_CURRENT))**2*SUBELEM.Ng[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig] 
                            
        return L2error
    
    
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
            self.ELMAT_file.write('NON_CUT_ELEMENTS\n')
        
        # INTEGRATE OVER THE SURFACE OF ELEMENTS WHICH ARE NOT CUT BY ANY INTERFACE (STANDARD QUADRATURES)
        print("     Integrate over non-cut elements...", end="")
        
        for ielem in self.NonCutElems: 
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]  
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.ng])
            if ELEMENT.Dom < 0:
                # MAP PSI VALUES FROM ELEMENT NODES TO GAUSS NODES
                PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                for ig in range(ELEMENT.ng):
                    SourceTermg[ig] = self.SourceTerm(ELEMENT.Xg[ig,:],PSIg[ig])
                    
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
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
        
        # INTEGRATE OVER THE SURFACES OF SUBELEMENTS IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate over cut-elements subelements...", end="")
        
        for ielem in self.CutElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]
            # NOW, EACH INTERFACE ELEMENT IS DIVIDED INTO SUBELEMENTS ACCORDING TO THE POSITION OF THE APPROXIMATED INTERFACE ->> TESSELLATION
            # ON EACH SUBELEMENT THE WEAK FORM IS INTEGRATED USING ADAPTED NUMERICAL INTEGRATION QUADRATURES
            ####### COMPUTE DOMAIN TERMS
            # LOOP OVER SUBELEMENTS 
            for SUBELEM in ELEMENT.SubElements:  
                # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
                SourceTermg = np.zeros([SUBELEM.ng])
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    for ig in range(SUBELEM.ng):
                        SourceTermg[ig] = self.SourceTerm(SUBELEM.Xg[ig,:],PSIg[ig])
                        
                # COMPUTE ELEMENTAL MATRICES
                if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
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
        
        # INTEGRATE OVER THE CUT EDGES IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate along cut-elements interface edges...", end="")
        
        for ielem in self.CutElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
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
        
        if self.VACUUM_VESSEL == self.COMPUTATIONAL:
            
            if self.ELMAT_output:
                self.ELMAT_file.write('BOUNDARY_ELEMENTS_INTERFACE\n')
            
            for elem in self.VacVessWallElems:
                # ISOLATE ELEMENT 
                ELEMENT = self.Elements[elem]
                # COMPUTE ELEMENTAL MATRICES
                if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
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
        
        print("Done!")   
        return
    
    def SolveSystem(self):
        # SOLVE LINEAR SYSTEM OF EQUATIONS AND OBTAIN PSI
        if self.PLASMA_BOUNDARY == "FIXE":
            self.PSI = np.zeros([self.Nn,1])
            LHSred, RHSred, plasmaboundnodes, unknownodes = self.StrongBCimposition()
            unknowns = np.linalg.solve(LHSred, RHSred)
            for inode, val in enumerate(unknowns):
                self.PSI[unknownodes[inode]] = val
            for node in plasmaboundnodes:
                self.PSI[node] = self.PSIAnalyticalSolution(self.X[node,:],self.PLASMA_CURRENT)
        else:
            self.PSI = np.linalg.solve(self.LHS, self.RHS)
        return
    
    def StrongBCimposition(self):
        # WE STRONGLY IMPOSE ANALYTICAL SOLUTION VALUES ON THE NODES FROM PLASMA BOUNDARY ELEMENTS
        RHS_temp = self.RHS.copy()
        # 1. FIND PLASMA BOUNDARY NODES
        plasmaboundnodes = set()
        for elem in self.PlasmaBoundElems:
            for node in self.Elements[elem].Te:
                plasmaboundnodes.add(node)
        plasmaboundnodes = np.array(list(plasmaboundnodes))
        # 2. STRONGLY IMPOSE BC BY PASSING LHS COLUMNS TO RHS
        unknownodes = list(range(self.Nn)) 
        for node in plasmaboundnodes:
            nodeval = self.PSIAnalyticalSolution(self.X[node,:],self.PLASMA_CURRENT)
            RHS_temp[:,0] -= nodeval*self.LHS[:,node]
            unknownodes.remove(node)
        # 3. REDUCE GLOBAL SYSTEM
        LHSred = self.LHS[np.ix_(unknownodes,unknownodes)]
        RHSred = RHS_temp[unknownodes,:]
        
        return LHSred, RHSred, plasmaboundnodes, unknownodes
    
    ##################################################################################################
    ############################################# OUTPUT #############################################
    ##################################################################################################
    
    def openOUTPUTfiles(self):
        if self.PSI_output:
            self.PSI_file = open(self.outputdir+'/UNKNO.dat', 'w')
            self.PSI_file.write('UNKNOWN_PSIpol_FIELD\n')
            
        if self.PSI_NORM_output:
            self.PSI_NORM_file = open(self.outputdir+'/PSIpol.dat', 'w')
            self.PSI_NORM_file.write('PSIpol_FIELD\n')
        
        if self.PSIcrit_output:
            self.PSIcrit_file = open(self.outputdir+'/PSIcrit.dat', 'w')
            self.PSIcrit_file.write('PSIcrit_VALUES\n')
        
        if self.PSI_B_output:
            self.PSI_B_file = open(self.outputdir+'/PSIpol_B.dat', 'w')
            self.PSI_B_file.write('PSIpol_B_VALUES\n')
        
        if self.RESIDU_output:
            self.RESIDU_file = open(self.outputdir+'/Residu.dat', 'w')
            self.RESIDU_file.write('RESIDU_VALUES\n')
        
        if self.ElementsClassi_output:
            self.ElementsClassi_file = open(self.outputdir+'/MeshElementsClassification.dat', 'w')
            self.ElementsClassi_file.write('MESH_ELEMENTS_CLASSIFICATION\n')
        
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file = open(self.outputdir+'/PlasmaBoundLS.dat', 'w')
            self.PlasmaLevSetVals_file.write('PLASMA_BOUNDARY_LEVEL_SET\n')
        
        if self.VacVessLevSetVals_output:
            self.VacVessLevSetVals_file = open(self.outputdir+'/VacuumVesselWallLS.dat', 'w')
            self.VacVessLevSetVals_file.write('VACUUM_VESSEL_LEVEL_SET\n')
        
        if self.ELMAT_output:
            self.ELMAT_file = open(self.outputdir+'/ElementalMatrices.dat', 'w')
            self.ELMAT_file.write('ELEMENTAL_MATRICES_FILE\n')
        
        return
    
    def closeOUTPUTfiles(self):
        if self.PSI_output:
            self.PSI_file.write('END_UNKNOWN_PSIpol_FIELD')
            self.PSI_file.close()
        
        if self.PSI_NORM_output:
            self.PSI_NORM_file.write('END_PSIpol_FIELD')
            self.PSI_NORM_file.close()
        
        if self.PSIcrit_output:
            self.PSIcrit_file.write('END_PSIcrit_VALUES')
            self.PSIcrit_file.close()
        
        if self.PSI_B_output:
            self.PSI_B_file.write('END_PSIpol_B_VALUES')
            self.PSI_B_file.close()
        
        if self.RESIDU_output:
            self.RESIDU_file.write('END_RESIDU_VALUES')
            self.RESIDU_file.close()
        
        if self.ElementsClassi_output:
            self.ElementsClassi_file.write('END_MESH_ELEMENTS_CLASSIFICATION')
            self.ElementsClassi_file.close()
        
        if self.VacVessLevSetVals_output:
            self.VacVessLevSetVals_file.write('END_VACUUM_VESSEL_LEVEL_SET')
            self.VacVessLevSetVals_file.close()
        
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file.write('END_PLASMA_BOUNDARY_LEVEL_SET')
            self.PlasmaLevSetVals_file.close()
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_ELEMENTAL_MATRICES_FILE\n')
            self.ELMAT_file.close()
            
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
        if self.PARAMS_output:
            self.PARAMS_file = open(self.outputdir+'/PARAMETERS.dat', 'w')
            self.PARAMS_file.write('SIMULATION_PARAMTERS_FILE\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('MESH_PARAMETERS\n')
            self.PARAMS_file.write("    NPOIN = {:d}\n".format(self.Nn))
            self.PARAMS_file.write("    NELEM = {:d}\n".format(self.Ne))
            self.PARAMS_file.write("    ELEM = {:d}\n".format(self.ElTypeALYA))
            self.PARAMS_file.write("    NBOUN = {:d}\n".format(self.Nbound))
            self.PARAMS_file.write('END_MESH_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('PROBLEM_TYPE_PARAMETERS\n')
            self.PARAMS_file.write("    PLASMA_BOUNDARY_equ = {:d}\n".format(self.PLASMA_BOUNDARY))
            self.PARAMS_file.write("    PLASMA_GEOMETRY_equ = {:d}\n".format(self.PLASMA_GEOMETRY))
            if self.PLASMA_CURRENT == 'FAKE':
                self.PARAMS_file.write("    PLASMA_CURRENT_equ = "+self.PLASMA_CURRENT )
            else:
                self.PARAMS_file.write("    PLASMA_CURRENT_equ = {:d}\n".format(self.PLASMA_CURRENT))
            self.PARAMS_file.write("    VACUUM_VESSEL_equ = {:d}\n".format(self.VACUUM_VESSEL))
            self.PARAMS_file.write("    TOTAL_PLASMA_CURRENT = {:f}\n".format(self.TOTAL_CURRENT))
            self.PARAMS_file.write('END_PROBLEM_TYPE_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write("    R0_equ = {:f}\n".format(self.R0))
            self.PARAMS_file.write("    EPSILON_equ = {:f}\n".format(self.epsilon))
            self.PARAMS_file.write("    KAPPA_equ = {:f}\n".format(self.kappa))
            self.PARAMS_file.write("    DELTA_equ = {:f}\n".format(self.delta))
            self.PARAMS_file.write('END_VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            if self.PLASMA_GEOMETRY == self.F4E_BOUNDARY:
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
            
            if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
                self.PARAMS_file.write('PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.PARAMS_file.write("    B0_equ = {:f}\n".format(self.B0))
                self.PARAMS_file.write("    q0_equ = {:f}\n".format(self.q0))
                self.PARAMS_file.write("    n_p_equ = {:f}\n".format(self.n_p))
                self.PARAMS_file.write("    g0_equ = {:f}\n".format(self.G0))
                self.PARAMS_file.write("    n_g_equ = {:f}\n".format(self.n_g))
                self.PARAMS_file.write('END_PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.PARAMS_file.write('\n')
            
            if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                self.PARAMS_file.write('EXTERNAL_COILS_PARAMETERS\n')
                self.PARAMS_file.write("    N_COILS_equ = {:d}\n".format(self.Ncoils))
                for COIL in self.COILS:
                    self.PARAMS_file.write("    Rposi = {:f}\n".format(COIL.X[0]))
                    self.PARAMS_file.write("    Zposi = {:f}\n".format(COIL.X[1]))
                    self.PARAMS_file.write("    Inten = {:f}\n".format(COIL.I))
                    self.PARAMS_file.write('\n')
                self.PARAMS_file.write('END_EXTERNAL_COILS_PARAMETERS\n')
                self.PARAMS_file.write('\n')
                
                self.PARAMS_file.write('EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.PARAMS_file.write("    N_SOLENOIDS_equ = {:d}\n".format(self.Nsolenoids))
                for SOLENOID in self.SOLENOIDS:
                    self.PARAMS_file.write("    Rlow = {:f}\n".format(SOLENOID.Xe[0,0]))
                    self.PARAMS_file.write("    Zlow = {:f}\n".format(SOLENOID.Xe[0,1]))
                    self.PARAMS_file.write("    Rup = {:f}\n".format(SOLENOID.Xe[1,0]))
                    self.PARAMS_file.write("    Zup = {:f}\n".format(SOLENOID.Xe[1,1]))
                    self.PARAMS_file.write("    Inten = {:f}\n".format(SOLENOID.I))
                    self.PARAMS_file.write('\n')
                self.PARAMS_file.write('END_EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('NUMERICAL_TREATMENT_PARAMETERS\n')
            self.PARAMS_file.write("    QUADRATURE_ORDER_equ = {:d}\n".format(self.QuadratureOrder))
            self.PARAMS_file.write("    MAX_EXT_IT_equ = {:d}\n".format(self.EXT_ITER))
            self.PARAMS_file.write("    EXT_TOL_equ = {:e}\n".format(self.EXT_TOL))
            self.PARAMS_file.write("    MAX_INT_IT_equ = {:d}\n".format(self.INT_ITER))
            self.PARAMS_file.write("    INT_TOL_equ = {:e}\n".format(self.INT_TOL))
            self.PARAMS_file.write("    Beta_equ = {:f}\n".format(self.beta))
            self.PARAMS_file.write('END_NUMERICAL_TREATMENT_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('END_SIMULATION_PARAMTERS_FILE\n')
            self.PARAMS_file.close()
        return
    
    def writePSI(self):
        if self.PSI_output:
            self.PSI_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PSI_file.write("{:d} {:e}\n".format(inode+1,float(self.PSI[inode])))
            self.PSI_file.write('END_ITERATION\n')
        return
    
    def writePSI_NORM(self):
        if self.PSI_NORM_output:
            self.PSI_NORM_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PSI_NORM_file.write("{:d} {:e}\n".format(inode+1,self.PSI_NORM[inode,0]))
            self.PSI_NORM_file.write('END_ITERATION\n')
        return
    
    def writePSI_B(self):
        if self.PSI_B_output:
            self.PSI_B_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.NnFW):
                self.PSI_B_file.write("{:d} {:e}\n".format(inode+1,self.PSI_B[inode,0]))
            self.PSI_B_file.write('END_ITERATION\n')
        return
    
    def writeresidu(self,which_loop):
        if self.RESIDU_output:
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
        if self.PSIcrit_output:
            self.PSIcrit_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,0,-1],self.Xcrit[1,0,0],self.Xcrit[1,0,1],self.PSI_0[0]))
            self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,1,-1],self.Xcrit[1,1,0],self.Xcrit[1,1,1],self.PSI_X[0]))
            self.PSIcrit_file.write('END_ITERATION\n')
        return
    
    def writeElementsClassification(self):
        if self.ElementsClassi_output:
            self.ElementsClassi_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            MeshClassi = self.ObtainClassification()
            for ielem in range(self.Ne):
                self.ElementsClassi_file.write("{:d} {:d}\n".format(ielem+1,MeshClassi[ielem]))
            self.ElementsClassi_file.write('END_ITERATION\n')
        return
    
    def writePlasmaBoundaryLS(self):
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PlasmaLevSetVals_file.write("{:d} {:e}\n".format(inode+1,self.PlasmaBoundLevSet[inode]))
            self.PlasmaLevSetVals_file.write('END_ITERATION\n')
        return
    
    def writeVacVesselBoundaryLS(self):
        if self.VacVessLevSetVals_output:
            for inode in range(self.Nn):
                self.VacVessLevSetVals_file.write("{:d} {:f}\n".format(inode+1,self.VacVessWallLevSet[inode]))
        return
    
    def writeconvergedsolution(self):
        if self.L2error_output:
            self.L2error_file = open(self.outputdir+'/PSIconverged.dat', 'w')
            self.L2error_file.write('PSI_CONVERGED_FIELD\n')
            for inode in range(self.Nn):
                self.L2error_file.write("{:d} {:e}\n".format(inode+1,self.PSI_CONV[inode])) 
            self.L2error_file.write('END_PSI_CONVERGED_FIELD\n')
            
            AnaliticalNorm = np.zeros([self.Nn])
            error = np.zeros([self.Nn])
            for inode in range(self.Nn):
                AnaliticalNorm[inode] = self.PSIAnalyticalSolution(self.X[inode,:],self.PLASMA_CURRENT)
                error[inode] = abs(AnaliticalNorm[inode]-self.PSI_CONV[inode])
                if error[inode] == 0:
                    error[inode] = 1e-15
                    
            self.L2error_file.write('ERROR_FIELD\n')
            for inode in range(self.Nn):
                self.L2error_file.write("{:d} {:e}\n".format(inode+1,error[inode])) 
            self.L2error_file.write('END_ERROR_FIELD\n')
            
            self.L2error_file.write("L2ERROR = {:e}".format(self.L2error))
    
            self.L2error_file.close()
        return
    
    
    ##################################################################################################
    ######################################## MAIN ALGORITHM ##########################################
    ##################################################################################################
    
    def EQUILI(self):
        # READ INPUT FILES
        print("READ INPUT FILES...")
        self.ReadMesh()
        self.ReadFixdata()
        self.ReadEQUILIdata()
        print('Done!')
        
        # OUTPUT RESULTS FOLDER
        # Check if the directory exists
        if not os.path.exists(self.outputdir):
            # Create the directory
            os.makedirs(self.outputdir)
        # COPY SIMULATION FILES
        self.copysimfiles()
        # WRITE SIMULATION PARAMETERS FILE (IF ON)
        self.writeparams() 
        # OPEN OUTPUT FILES
        self.openOUTPUTfiles()    
                
        # INITIALIZATION
        print("INITIALIZATION...")
        self.it = 0
        self.it_EXT = 0
        self.it_INT = 0
        self.Initialization()
        print('Done!')

        if self.plotPSI_output:
            self.PlotSolutionPSI()  # PLOT INITIAL SOLUTION

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.converg_EXT = False
        self.it_EXT = 0
        
        #######################################################
        ################## EXTERNAL LOOP ######################
        #######################################################
        while (self.converg_EXT == False and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = False
            self.it_INT = 0
            #######################################################
            ################## INTERNAL LOOP ######################
            #######################################################
            while (self.converg_INT == False and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                
                # WRITE ITERATION CONFIGURATION
                self.writePlasmaBoundaryLS()
                self.writeElementsClassification()
                if self.plotElemsClassi_output:
                    self.PlotClassifiedElements()
                    
                # INNER LOOP ALGORITHM: SOLVING GRAD-SHAFRANOV BVP WITH CutFEM
                self.AssembleGlobalSystem()                 # 0. ASSEMBLE CUTFEM SYSTEM
                self.SolveSystem()                          # 1. SOLVE CutFEM SYSTEM  ->> PSI
                self.writePSI()                             #    WRITE SOLUTION 
                self.ComputeCriticalPSI(self.PSI)           # 2. COMPUTE CRITICAL VALUES   PSI_0 AND PSI_X
                self.writePSIcrit()                         #    WRITE CRITICAL POINTS
                self.NormalisePSI()                         # 3. NORMALISE PSI RESPECT TO CRITICAL VALUES  ->> PSI_NORM 
                self.writePSI_NORM()                        #    WRITE NORMALISED SOLUTION
                if self.plotPSI_output:
                    self.PlotSolutionPSI()                  #    PLOT SOLUTION AND NORMALISED SOLUTION
                self.CheckConvergence('PSI_NORM')           # 4. CHECK CONVERGENCE OF PSI_NORM FIELD
                self.writeresidu("INTERNAL")                #    WRITE INTERNAL LOOP RESIDU
                self.UpdateElements()                       # 5. UPDATE MESH ELEMENTS CLASSIFACTION RESPECT TO NEW PLASMA BOUNDARY LEVEL-SET
                self.UpdatePSI('PSI_NORM')                  # 6. UPDATE PSI_NORM VALUES (PSI_NORM[:,0] = PSI_NORM[:,1])
                self.UpdateElementalPSI()                   # 7. UPDATE PSI_NORM VALUES IN CORRESPONDING ELEMENTS (ELEMENT.PSIe = PSI_NORM[ELEMENT.Te,0])
                self.UpdateElementalPSIg(self.PLASMAbound)  # 8. UPDATE ELEMENTAL CONSTRAINT VALUES PSIgseg FOR PLASMA/VACUUM INTERFACE
                
                #######################################################
                ################ END INTERNAL LOOP ####################
                #######################################################
            
            self.ComputeTotalPlasmaCurrentNormalization()
            print('COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
            self.PSI_B[:,1] = self.ComputePSI_B()     # COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B WITH INTERNALLY CONVERGED PSI_NORM
            print('Done!')
            
            self.CheckConvergence('PSI_B')            # CHECK CONVERGENCE OF VACUUM VESSEL FIEST WALL PSI VALUES  (PSI_B)
            self.writeresidu("EXTERNAL")              # WRITE EXTERNAL LOOP RESIDU 
            self.UpdatePSI('PSI_B')                   # UPDATE PSI_NORM AND PSI_B VALUES
            self.UpdateElementalPSIg(self.VACVESbound)   # UPDATE ELEMENTAL CONSTRAINT VALUES PSIgseg FOR VACUUM VESSEL FIRST WALL INTERFACE 
            
            #######################################################
            ################ END EXTERNAL LOOP ####################
            #######################################################
            
        print('SOLUTION CONVERGED')
        self.PlotSolutionPSI()
        
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY and self.PLASMA_CURRENT != self.PROFILES_CURRENT:
            self.L2error = self.ComputeL2error()
            self.writeconvergedsolution()
        
        self.closeOUTPUTfiles()
        return
    
    ##################################################################################################
    ############################### RENDERING AND REPRESENTATION #####################################
    ##################################################################################################
    
    def PlotFIELD(self,FIELD):
        
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        axs.set_xlim(self.Xmin,self.Xmax)
        axs.set_ylim(self.Ymin,self.Ymax)
        a = axs.tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], FIELD[self.activenodes], levels=30)
        axs.tricontour(self.X[self.activenodes,0],self.X[self.activenodes,1], FIELD[self.activenodes], levels=[0], colors = 'black')
        axs.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        if self.VACUUM_VESSEL == self.FIRST_WALL:
            axs.tricontour(self.X[:,0],self.X[:,1], self.VacVessWallLevSet, levels=[0], colors = 'orange')
        plt.colorbar(a, ax=axs)
        plt.show()
        
        return

    
    def PlotSolutionPSI(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PSI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, 
        AND PSI_NORM IF NORMALISED. """
        
        def subplotfield(self,ax,field):
            a = ax.tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], field[self.activenodes], levels=50)
            ax.tricontour(self.X[:,0],self.X[:,1], field, levels=[0], colors = 'black')
            ax.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
            if self.VACUUM_VESSEL == self.FIRST_WALL:
                ax.tricontour(self.X[:,0],self.X[:,1], self.VacVessWallLevSet, levels=[0], colors = 'orange')
            ax.set_xlim(self.Xmin, self.Xmax)
            ax.set_ylim(self.Ymin, self.Ymax)
            plt.colorbar(a, ax=ax)
            return
        
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            psi_sol = " normalised solution PSI_NORM"
        else:
            psi_sol = " solution PSI"
        
        if self.it == 0:  # INITIAL GUESS PLOT
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI_NORM[:,0])
            axs.set_title('Initial PSI guess')
            plt.show(block=False)
            plt.pause(0.8)
            
        elif self.converg_EXT:  # CONVERGED SOLUTION PLOT
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI_CONV)
            axs.set_title('Converged'+psi_sol)
            plt.show()
            
        elif self.PLASMA_CURRENT == self.PROFILES_CURRENT:  # ITERATION SOLUTION FOR PROFILES PLASMA CURRENT (PLOT PSI and PSI_NORM)
            fig, axs = plt.subplots(1, 2, figsize=(11,5))
            # LEFT PLOT: PSI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
            subplotfield(self,axs[0],self.PSI[:,0])
            axs[0].set_title('Poloidal magnetic flux PSI')
            # RIGHT PLOT: NORMALISED PSI at iteration N+1
            subplotfield(self,axs[1],self.PSI_NORM[:,1])
            axs[1].set_title('Normalised poloidal magnetic flux PSI_NORM')
            axs[1].yaxis.set_visible(False)
            ## PLOT LOCATION OF CRITICAL POINTS
            for i in range(2):
                # LOCAL EXTREMUM
                axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'x',color='red', s = 40, linewidths = 2)
                # SADDLE POINT
                axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'x',color='green', s = 40, linewidths = 2)
            plt.suptitle("Iteration n = "+str(self.it))
            plt.show(block=False)
            plt.pause(0.8)
                
        else:  # ITERATION SOLUTION FOR ANALYTICAL PLASMA CURRENT CASES (PLOT PSI)
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI[:,0])
            axs.set_title('Poloidal magnetic flux PSI')
            axs.set_title("Iteration n = "+str(self.it)+ psi_sol)
            plt.show(block=False)
            plt.pause(0.8)
            
        return
    
    def InspectElement(self,element_index,PSI,TESSELLATION,BOUNDARY,NORMALS,QUADRATURE):
        
        ELEM = self.Elements[element_index]
        Xmin = np.min(ELEM.Xe[:,0])-0.1
        Xmax = np.max(ELEM.Xe[:,0])+0.1
        Ymin = np.min(ELEM.Xe[:,1])-0.1
        Ymax = np.max(ELEM.Xe[:,1])+0.1
        if ELEM.ElType == 1:
            numedges = 3
        elif ELEM.ElType == 2:
            numedges = 4
            
        color = self.ElementColor(ELEM.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

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
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.PlasmaLSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEM.Xe[inode,0],ELEM.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            for INTERFACE in ELEM.InterfApprox:
                axs[1].scatter(INTERFACE.Xint[:,0],INTERFACE.Xint[:,1],marker='o',color='red',s=100, zorder=5)
                for SEGMENT in INTERFACE.Segments:
                    axs[1].scatter(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for INTERFACE in ELEM.InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                    dl = 10
                    axs[1].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2 and self.VACUUM_VESSEL == self.COMPUTATIONAL:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder = 5)
            else:
                for isub, SUBELEM in enumerate(ELEM.SubElements):
                    # PLOT QUADRATURE INTEGRATION POINTS
                    axs[1].scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c=colorlist[isub], zorder=3)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder=5)
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
            for i in range(self.numedges):
                plt.plot([self.X[self.T[e,i],0], self.X[self.T[e,int((i+1)%self.n)],0]], 
                        [self.X[self.T[e,i],1], self.X[self.T[e,int((i+1)%self.n)],1]], color='black', linewidth=1)
        plt.show()
        return
    
    def PlotClassifiedElements(self):
        plt.figure(figsize=(5,6))
        plt.ylim(self.Ymin-0.25,self.Ymax+0.25)
        plt.xlim(self.Xmin-0.25,self.Xmax+0.25)
        
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'black')
        # PLOT PLASMA/VACUUM INTERFACE ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gold')
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numvertices+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
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
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        
        axs[0].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        axs[0].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[1].set_xlim(6,6.4)
        if self.PLASMA_GEOMETRY == self.FIRST_WALL:
            axs[1].set_ylim(2.5,3.5)
        elif self.PLASMA_GEOMETRY == self.F4E_BOUNDARY:
            axs[1].set_ylim(2.2,2.6)

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
                for INTERFACE in self.Elements[elem].InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        # PLOT INTERFACE APPROXIMATIONS
                        axs[0].plot(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1], linestyle='-',color = 'red', linewidth = 2)
                        axs[1].plot(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1], linestyle='-',marker='o',color = 'red', linewidth = 2)
                        # PLOT NORMAL VECTORS
                        Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                        axs[i].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
                
        axs[1].set_aspect('equal')
        plt.show()
        return
    
    
    def PlotInterfaceValues(self):
        """ Function which plots the values PSIgseg at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """

        # COLLECT PSIg DATA ON PLASMA BOUNDARY
        X_Dg = np.zeros([self.NnPB,self.dim])
        PSI_Dg = np.zeros([self.NnPB])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PSI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for elem in self.PlasmaBoundElems:
            for INTERFACE in self.Elements[elem].InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    for inode in range(SEGMENT.ng):
                        X_Dg[k,:] = SEGMENT.Xg[inode,:]
                        PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                        k += 1
            for node in range(self.Elements[elem].n):
                X_D[l,:] = self.Elements[elem].Xe[node,:]
                PSI_D[l] = self.PSI[self.Elements[elem].Te[node]]
                l += 1
            
        # COLLECT PSIg DATA ON VACUUM VESSEL FIRST WALL 
        X_Bg = np.zeros([self.NnFW,self.dim])
        PSI_Bg = np.zeros([self.NnFW])
        X_B = np.zeros([len(self.VacVessWallElems)*self.n,self.dim])
        PSI_B = np.zeros([len(self.VacVessWallElems)*self.n])
        k = 0
        l = 0
        for elem in self.VacVessWallElems:
            for INTERFACE in self.Elements[elem].InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    for inode in range(SEGMENT.ng):
                        X_Bg[k,:] = SEGMENT.Xg[inode,:]
                        PSI_Bg[k] = SEGMENT.PSIgseg[inode]
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
        
        # COLLECT PSIgseg DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D,self.dim])
        PSI_Dexact = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        PSI_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PSI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        error = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for elem in self.PlasmaBoundElems:
            for INTERFACE in range(self.Elements[elem].InterfApprox):
                for SEGMENT in INTERFACE.Segments:
                    for inode in range(SEGMENT.ng):
                        X_Dg[k,:] = SEGMENT.Xg[inode,:]
                        if self.PLASMA_CURRENT != self.PROFILES_CURRENT:
                            PSI_Dexact[k] = self.PSIAnalyticalSolution(X_Dg[k,:],self.PLASMA_CURRENT)
                        else:
                            PSI_Dexact[k] = SEGMENT.PSIgseg[inode]
                        PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                        k += 1
            for jnode in range(self.Elements[elem].n):
                X_D[l,:] = self.Elements[elem].Xe[jnode,:]
                PSI_Dexact_node = self.PSIAnalyticalSolution(X_D[l,:],self.PLASMA_CURRENT)
                PSI_D[l] = self.PSI[self.Elements[elem].Te[jnode]]
                error[l] = np.abs(PSI_D[l]-PSI_Dexact_node)
                l += 1
            
        fig, axs = plt.subplots(1, 4, figsize=(18,6)) 
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
        
        axs[3].set_aspect('equal')
        axs[3].set_ylim(self.Ymin-0.5,self.Ymax+0.5)
        axs[3].set_xlim(self.Xmin-0.5,self.Xmax+0.5)
        norm = plt.Normalize(np.log(error).min(),np.log(error).max())
        linecolors_error = cmap(norm(np.log(error)))
        axs[3].scatter(X_D[:,0],X_D[:,1],color = linecolors_error)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[3])

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
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gray', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            
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
                plt.scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c='gold')
            # PLOT INTERFACE LINEAR APPROXIMATION AND INTEGRATION POINTS
            for INTERFACE in range(ELEMENT.InterfApprox):
                for SEGMENT in INTERFACE.Segments:
                    # PLOT INTERFACE LINEAR APPROXIMATION
                    plt.plot(SEGMENT.Xseg[:,0], SEGMENT.Xseg[:,1], color='green', linewidth=1)
                    # PLOT INTERFACE QUADRATURE
                    plt.scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='o',c='green')
                
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='darkturquoise', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            if self.VACUUM_VESSEL == self.COMPUTATIONAL:
                plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='darkturquoise')
            else:
                for SUBELEM in ELEMENT.SubElements:
                    # PLOT SUBELEMENT EDGES
                    for i in range(self.n):
                        plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='darkturquoise', linewidth=1)
                    # PLOT QUADRATURE INTEGRATION POINTS
                    plt.scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c='darkturquoise')
            # PLOT INTERFACE LINEAR APPROXIMATION AND INTEGRATION POINTS
            for INTERFACE in range(ELEMENT.InterfApprox):
                for SEGMENT in INTERFACE.Segments:
                    # PLOT INTERFACE LINEAR APPROXIMATION
                    plt.plot(SEGMENT.Xseg[:,0], SEGMENT.Xseg[:,1], color='orange', linewidth=1)
                    # PLOT INTERFACE QUADRATURE
                    plt.scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='o',c='orange')

        plt.show()
        return

    
    def PlotError(self):
        self.ComputePlasmaNodes()
        
        AnaliticalNorm = np.zeros([self.Nn])
        for inode in range(self.Nn):
            AnaliticalNorm[inode] = self.PSIAnalyticalSolution(self.X[inode,:],self.PLASMA_CURRENT)
           
        error = np.zeros([len(self.plasmanodes)])
        for i, inode in enumerate(self.plasmanodes):
            error[i] = abs(AnaliticalNorm[inode]-self.PSI_CONV[inode])
            if error[i] == 0:
                error[i] = 1e-15
            
        print(np.linalg.norm(error))
            
        fig, axs = plt.subplots(1, 3, figsize=(16,5))
        axs[0].set_xlim(self.Xmin,self.Xmax)
        axs[0].set_ylim(self.Ymin,self.Ymax)
        a = axs[0].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], AnaliticalNorm[self.activenodes], levels=30)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].tricontour(self.X[self.activenodes,0],self.X[self.activenodes,1], AnaliticalNorm[self.activenodes], levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Xmin,self.Xmax)
        axs[1].set_ylim(self.Ymin,self.Ymax)
        a = axs[1].tricontourf(self.X[self.activenodes,0],self.X[self.activenodes,1], self.PSI_CONV[self.activenodes], levels=30)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PSI_CONV, levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[1])

        axs[2].set_xlim(self.Xmin,self.Xmax)
        axs[2].set_ylim(self.Ymin,self.Ymax)
        a = axs[2].tricontourf(self.X[self.plasmanodes,0],self.X[self.plasmanodes,1], np.log(error), levels=30)
        axs[2].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        plt.colorbar(a, ax=axs[2])

        plt.show()
        
        return
    
    @staticmethod
    def ElementColor(dom):
        if dom == -1:
            color = 'red'
        elif dom == 0:
            color = 'gold'
        elif dom == 1:
            color = 'grey'
        elif dom == 2:
            color = 'cyan'
        elif dom == 3:
            color = 'black'
        return color
    
    def PlotREFERENCE_PHYSICALelement(self,element_index,TESSELLATION,BOUNDARY,NORMALS,QUADRATURE):
        ELEM = self.Elements[element_index]
        Xmin = np.min(ELEM.Xe[:,0])-0.1
        Xmax = np.max(ELEM.Xe[:,0])+0.1
        Ymin = np.min(ELEM.Xe[:,1])-0.1
        Ymax = np.max(ELEM.Xe[:,1])+0.1
        if ELEM.ElType == 1:
            numedges = 3
        elif ELEM.ElType == 2:
            numedges = 4
            
        color = self.ElementColor(ELEM.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        XIe = ReferenceElementCoordinates(ELEM.ElType,ELEM.ElOrder)
        XImin = np.min(XIe[:,0])-0.4
        XImax = np.max(XIe[:,0])+0.25
        ETAmin = np.min(XIe[:,1])-0.4
        ETAmax = np.max(XIe[:,1])+0.25
        axs[0].set_xlim(XImin,XImax)
        axs[0].set_ylim(ETAmin,ETAmax)
        axs[0].tricontour(XIe[:,0],XIe[:,1], ELEM.PlasmaLSe, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[0].plot([XIe[iedge,0],XIe[int((iedge+1)%ELEM.numedges),0]],[XIe[iedge,1],XIe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.PlasmaLSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[0].scatter(XIe[inode,0],XIe[inode,1],s=120,color=cl,zorder=5)

        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[0].plot([SUBELEM.XIe[i,0], SUBELEM.XIe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.XIe[i,1], SUBELEM.XIe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[0].scatter(SUBELEM.XIe[:,0],SUBELEM.XIe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            for INTERFACE in ELEM.InterfApprox:
                axs[0].scatter(INTERFACE.XIint[:,0],INTERFACE.XIint[:,1],marker='o',color='red',s=100, zorder=5)
                for SEGMENT in INTERFACE.Segments:
                    axs[0].scatter(SEGMENT.XIseg[:,0],SEGMENT.XIseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for INTERFACE in ELEM.InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                    dl = 10
                    #axs[0].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEM.XIg[:,0],ELEM.XIg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2 and self.VACUUM_VESSEL == self.COMPUTATIONAL:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEM.XIg[:,0],ELEM.XIg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[0].scatter(SEGMENT.XIg[:,0],SEGMENT.XIg[:,1],marker='x',color='grey',s=50, zorder = 5)
            else:
                for isub, SUBELEM in enumerate(ELEM.SubElements):
                    # PLOT QUADRATURE INTEGRATION POINTS
                    axs[0].scatter(SUBELEM.XIg[:,0],SUBELEM.XIg[:,1],marker='x',c=colorlist[isub], zorder=3)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[0].scatter(SEGMENT.XIg[:,0],SEGMENT.XIg[:,1],marker='x',color='grey',s=50, zorder=5)
                        
                        
        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.PlasmaLSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEM.Xe[inode,0],ELEM.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            for INTERFACE in ELEM.InterfApprox:
                axs[1].scatter(INTERFACE.Xint[:,0],INTERFACE.Xint[:,1],marker='o',color='red',s=100, zorder=5)
                for SEGMENT in INTERFACE.Segments:
                    axs[1].scatter(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for INTERFACE in ELEM.InterfApprox:
                for SEGMENT in INTERFACE.Segments:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                    dl = 10
                    axs[1].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2 and self.VACUUM_VESSEL == self.COMPUTATIONAL:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder = 5)
            else:
                for isub, SUBELEM in enumerate(ELEM.SubElements):
                    # PLOT QUADRATURE INTEGRATION POINTS
                    axs[1].scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c=colorlist[isub], zorder=3)
                # PLOT INTERFACE INTEGRATION POINTS
                for INTERFACE in ELEM.InterfApprox:
                    for SEGMENT in INTERFACE.Segments:
                        axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder=5)
                
        return
    