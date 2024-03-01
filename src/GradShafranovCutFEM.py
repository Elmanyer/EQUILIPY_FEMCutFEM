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
        self.PLASMA_BOUNDARY = None
        self.CASE = None                    # CASE SOLUTION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.InterElems = None              # LIST OF CUT ELEMENTS (INDEXES), CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.BoundaryElems = None           # LIST OF ELEMENTS (INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        
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
        self.INT_TOL = None                      # INTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.EXT_TOL = None                      # EXTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.INT_ITER = None                     # INTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.EXT_ITER = None                     # EXTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.converg_EXT = None                  # EXTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.converg_INT = None                  # INTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.it_EXT = None                       # EXTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self_it_INT = None                       # INTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it = 0                              # TOTAL NUMBER OF ITERATIONS COUNTER
        self.alpha = None                        # AIKTEN'S SCHEME RELAXATION CONSTANT
        
        # PARAMETERS FOR COILS
        self.Ncoils = None              # TOTAL NUMBER OF COILS
        self.Xcoils = None              # COILS' COORDINATE MATRIX 
        self.Icoils = None              # COILS' CURRENT
        
        # PLASMA REGION GEOMETRY
        self.epsilon = None                 # PLASMA REGION ASPECT RATIO
        self.kappa = None                   # PLASMA REGION ELONGATION
        self.delta = None                   # PLASMA REGION TRIANGULARITY
        self.Rmax = None                    # PLASMA REGION MAJOR RADIUS
        self.Rmin = None                    # PLASMA REGION MINOR RADIUS
        self.R0 = None                      # PLASMA REGION MEAN RADIUS
        
        self.beta = 1e8                     # NITSCHE'S METHOD PENALTY TERM
        self.coeffs = []                    # ANALYTICAL SOLUTION/INITIAL GUESS COEFFICIENTS
        
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
                    elif l[0] == 'Xposi:':    # READ i-th COIL X POSITION
                        self.Xcoils[i,0] = float(l[1])
                    elif l[0] == 'Yposi:':    # READ i-th COIL Y POSITION
                        self.Xcoils[i,1] = float(l[1])
                    elif l[0] == 'Inten:':    # READ i-th COIL INTENSITY
                        self.Icoils[i] = float(l[1])
                        i += 1
                    
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
        
        # 1. INTERPOLATE PHI VALUES ON A FINER STRUCTURED MESH USING PHI ON NODES
        # DEFINE FINER STRUCTURED MESH
        Mr = 60
        Mz = 80
        rfine = np.linspace(np.min(self.X[:,0]), np.max(self.X[:,0]), Mr)
        zfine = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), Mz)
        # INTERPOLATE PHI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine)
        PHIfine = griddata((self.X[:,0],self.X[:,1]), PHI, (Rfine, Zfine), method='cubic')
        
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
        
        PHI_0 = 0
        PHI_X = 0
        if nature == "LOCAL EXTREMUM":
            # FOR THE MAGNETIC AXIS VALUE PHI_0, THE LOCAL EXTREMUM SHOULD LIE INSIDE THE PLASMA REGION
            elem = SearchElement(self.Elements,Xcrit,self.PlasmaElems)
            PHI_0 = self.Elements[elem].ElementalInterpolation(Xcrit,PHI[self.Elements[elem].Te])
        else:
            elem = SearchElement(self.Elements,Xcrit,self.VacuumElems)
            PHI_X = self.Elements[elem].ElementalInterpolation(Xcrit,PHI[self.Elements[elem].Te])

        return PHI_0, PHI_X
    
    
    def NormalisePHI(self,PHI):
        
        PHI_0, PHI_X = self.ComputeCriticalPHI(PHI)
        
        PHIbar = np.zeros([self.Nn])
        for i in range(self.Nn):
            PHIbar[i] = (PHI[i]-PHI_X)/np.abs(PHI_0-PHI_X)
        
        return PHIbar
    
    
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
        self.PHI_inner1 = np.linalg.solve(self.LHS, self.RHS)
        
        # NORMALISE PHI ACCORDING TO OBTAINED PHI_0 AND PHI_X
        PHIbar = self.NormalisePHI(self.PHI_inner1[:,0])
        self.PHI_inner1 = np.zeros([self.Nn,1])
        for node in range(self.Nn):
            self.PHI_inner1[node] = PHIbar[node]
            
        self.PHI[:,self.it] = self.PHI_inner1[:,0]
        
        return
    
    
    ##################################################################################################
    ############################### CONVERGENCE VALIDATION ###########################################
    ##################################################################################################
    
    def CheckConvergence(self,loop):
        
        if loop == "INTERNAL":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_inner1) > 0:
                L2residu = np.linalg.norm(self.PHI_inner1 - self.PHI_inner0)/np.linalg.norm(self.PHI_inner1)
            else: 
                L2residu = np.linalg.norm(self.PHI_inner1 - self.PHI_inner0)
            if L2residu < self.INT_TOL:
                self.converg_INT = False   # STOP WHILE LOOP 
                self.PHI_outer1 = self.PHI_inner1
            else:
                self.converg_INT = True
                self.PHI_inner0 = self.PHI_inner1
            
        elif loop == "EXTERNAL":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_outer1) > 0:
                L2residu = np.linalg.norm(self.PHI_outer1 - self.PHI_outer0)/np.linalg.norm(self.PHI_outer1)
            else: 
                L2residu = np.linalg.norm(self.PHI_outer1 - self.PHI_outer0)
            if L2residu < self.EXT_TOL:
                self.converg_EXT = False   # STOP WHILE LOOP 
                self.PHI = self.PHI[:,:self.it+1]
                self.PHI_converged = self.PHI_outer1
            else:
                self.converg_EXT = True
                self.PHI_outer0 = self.PHI_outer1
        return 
    
    def UpdateElementalPHI(self):
        for element in self.Elements:
            element.PHIe = self.PHI_inner1[element.Te]
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
            -> FREE-BOUNDARY PROBLEM: """
            
        if self.PLASMA_BOUNDARY == 'FIXED':
            # ADIMENSIONALISE MESH
            Xstar = self.X/self.R0
            LS0 = np.zeros([self.Nn])
            coeffs = self.ComputeLinearSolutionCoefficients()
            for i in range(self.Nn):
                LS0[i] = Xstar[i,0]**4/8 + coeffs[0] + coeffs[1]*Xstar[i,0]**2 + coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
                
        elif self.PLASMA_BOUNDARY == 'FREE':
            LS0 = np.zeros([self.Nn])
            
        return LS0
    
    def InitialiseElements(self):
        """ Function initialising attribute ELEMENTS which is a list of all elements in the mesh. """
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.LevelSet[self.T[e,:]],self.PHI_inner0[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    def InitialisePHI(self):  
        """ INITIALISE PHI VECTORS WHERE THE DIFFERENT SOLUTIONS WILL BE STORED ITERATIVELY DURING THE SIMULATION AND COMPUTE INITIAL GUESS."""
        # INITIALISE PHI VECTORS
        self.PHI_INT = np.zeros([self.Nn,2])      # INTERNAL LOOP SOLUTION AT INTARTIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PHI_EXT = np.zeros([self.Nn,2])      # EXTERNAL LOOP SOLUTION AT INTARTIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PHI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION 
        # COMPUTE INITIAL GUESS AND STORE IT FOR BOTH INTERNAL AND EXTERNAL SOLUTIONS FOR N=0
        PHI0 = self.InitialGuess()
        self.PHI_INT[:,0] = PHI0     
        self.PHI_EXT[:,0] = PHI0
        return
    
    
    def Initialization(self):
        """ Routine which initialises all the necessary elements in the problem """
        # INITIALISE VARIABLES
        print("     -> COMPUTE INITIAL GUESS...", end="")
        self.InitialisePHI()
        print('Done!')

        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.LevelSet = self.InitialLevelSet()
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
    
    def ComputeBoundaryPHI(self):
        
        """ FUNCTION TO COMPUTE THE COMPUTATIONAL DOMAIN BOUNDARY VALUES FOR PHI, PHI_B. 
        THESE MUST BE TREATED AS NATURAL BOUNDARY CONDITIONS (DIRICHLET BOUNDARY CONDITIONS).
        SUCH VALUES ARE OBTAINED BY ACCOUNTING FOR THE CONTRIBUTIONS FROM THE EXTERNAL
        FIXED COILS AND THE CONTRIBUTION FROM THE PLASMA CURRENT ITSELF, FOR WHICH WE 
        INTEGRATE THE PLASMA'S GREEN FUNCTION.

        IN ORDER TO SOLVE A FIXED VALUE PROBLEM INSIDE THE COMPUTATIONAL DOMAIN, THE BOUNDARY VALUES
        PHI_B MUST COMPUTED FROM THE PREVIOUS SOLUTION FOR PHIPOL.  """
        
        # FOR FIXED PLASMA BOUNDARY, THE COMPUTATIONAL DOMAIN BOUNDARY PHI VALUES ARE IRRELEVANT, AS THE PLASMA REGION SHAPE IS ALREADY DEFINED
        if self.PLASMA_BOUNDARY == 'FIXED':  
            for elem in self.BoundaryElems:
                self.Elements[elem].PHI_Be = np.zeros([self.Elements[elem].Nebound,self.n])
        
        # FOR FREE PLASMA BOUNDARY, THE COMPUTATIONAL DOMAIN BOUNDARY PHI VALUES ARE COMPUTED FROM THE GRAD-SHAFRANOV OPERATOR'S GREEN FUNCTION
        elif self.PLASMA_BOUNDARY == 'FREE':  
            for elem in self.BoundaryElems:
                self.Elements[elem].ComputeElementalPHI_B(self.Elements,self.PlasmaElems,self.InterElems,self.Jphi,self.Ncoils,self.Xcoils,self.Icoils)
            
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
        self.converg_EXT = True
        self.it_EXT = 0
        self.it = 0
        while (self.converg_EXT == True and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = True
            self.it_INT = 0
            while (self.converg_INT == True and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                self.ComputeBoundaryPHI()
                self.AssembleGlobalSystem()
                self.SolveSystem()
                self.UpdateElementalPHI()
                self.CheckConvergence("INTERNAL")
                
            self.CheckConvergence("EXTERNAL")

        self.PlotSolution(self.PHI_converged,colorbar=True)
        
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