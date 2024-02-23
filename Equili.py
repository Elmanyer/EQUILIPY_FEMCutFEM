""" This script contains the Python object defining a plasma equilibrium problem, modeled using the Grad-Shafranov equation
in an axisymmetrical system such as a tokamak. """

import numpy as np
import matplotlib.pyplot as plt
from random import random
from scipy.interpolate import griddata
from GaussQuadrature import *
from ShapeFunctions import *
from ElementObject import *

class Equili:
    
    # GENERAL PARAMETERS
    epsilon0 = 8.8542E-12       # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19               # J eV-1   Botlzmann constant

    def __init__(self,folder_loc,ElementType,ElementOrder,CASE):
        self.directory = folder_loc
        self.case = folder_loc[folder_loc.rfind("/")+1:]
        
        # DECLARE ATTRIBUTES
        self.ElType = ElementType           # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = ElementOrder         # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC
        self.epsilon = None                 # PLASMA REGION ASPECT RATIO
        self.kappa = None                   # PLASMA REGION ELONGATION
        self.delta = None                   # PLASMA REGION TRIANGULARITY
        self.Rmax = None                    # PLASMA REGION MAJOR RADIUS
        self.Rmin = None                    # PLASMA REGION MINOR RADIUS
        self.R0 = None                      # PLASMA REGION MEAN RADIUS
        self.QuadratureOrder = 2            # NUMERICAL INTEGRATION QUADRATURE ORDER 
        self.TOL_inner = 1e-3               # INNER LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.TOL_outer = 1e-3               # OUTER LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.it = 0                         # ITERATIONS COUNTER
        self.itmax = 5                      # LOOP STRUCTURES MAXIMUM ITERATIONS NUMBER
        self.beta = 1e8                     # NITSCHE'S METHOD PENALTY TERM
        self.CASE = CASE                    # CASE SOLUTION
        self.coeffs = []                    # ANALYTICAL SOLUTION COEFFICIENTS

        return
    
    def ReadMesh(self):
        # NUMBER OF NODES PER ELEMENT
        self.n = ElementalNumberOfNodes(self.ElType, self.ElOrder)
        
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
    
    def AnalyticalSolution(self,X):
        """ Function which computes the analytical solution (if it exists) at point with coordinates X. """
        # DIMENSIONALESS COORDINATES
        Xstar = X/self.R0
        #Xstar = X
        
        if self.CASE == 'LINEAR':
            if not self.coeffs: 
                self.coeffs = self.ComputeLinearSolutionCoefficients()  # [D1, D2, D3]
            PHIexact = (Xstar[0]**4)/8 + self.coeffs[0] + self.coeffs[1]*Xstar[0]**2 + self.coeffs[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
            
        elif self.CASE == 'NONLINEAR':
            if not self.coeffs:
                self.coeffs = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0]
            PHIexact = np.sin(self.coeffs[0]*(Xstar[0]+self.coeffs[2]))*np.cos(self.coeffs[1]*Xstar[1])
            
        return PHIexact
    
    def Jphi(self,R,Z,phi):
        """ Function which computes the plasma current source term on the right hand side of the Grad-Shafranov equation. """
        R = R/self.R0
        Z = Z/self.R0
        
        if self.CASE == 'LINEAR':
            # self.coeffs = [D1 D2 D3]  for linear solution
            jphi = R/self.mu0
            
        if self.CASE == 'NONLINEAR': 
            # self.coeffs = [Kr Kz R0]  for nonlinear solution
            jphi = -((self.coeffs[0]**2+self.coeffs[1]**2)*phi+(self.coeffs[0]/R)*np.cos(self.coeffs[0]*(R+self.coeffs[2]))*np.cos(self.coeffs[1]*Z)
            +R*((np.sin(self.coeffs[0]*(R+self.coeffs[2]))*np.cos(self.coeffs[1]*Z))**2-phi**2+np.exp(-np.sin(self.coeffs[0]*(R+self.coeffs[2]))*
                                                                                            np.cos(self.coeffs[1]*Z))-np.exp(-phi)))/(self.mu0*R)
        return jphi
    
    
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
    
    
    def InitialGuess(self):
        """ This function computes the problem's initial guess. 
                - PHI0: initial guess values  
                """
        PHI0 = np.zeros([self.Nn])
        for i in range(self.Nn):
            PHIexact = self.AnalyticalSolution(self.X[i,:])
            PHI0[i] = PHIexact*2*random()
            
        return PHI0
    
    
    def InitialLevelSet(self):
        """ Use the analytical solution for the LINEAR case as initial Level-Set function. The plasma region is characterised by a negative value of Level-Set. """
        # ADIMENSIONALISE MESH
        Xstar = self.X/self.R0
        LS0 = np.zeros([self.Nn])
        coeffs = self.ComputeLinearSolutionCoefficients()
        for i in range(self.Nn):
            LS0[i] = Xstar[i,0]**4/8 + coeffs[0] + coeffs[1]*Xstar[i,0]**2 + coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
        return LS0
    
    
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
            LSe = self.Elements[e].LSe  # elemental nodal level-set values
            for i in range(self.n-1):
                if np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between nodal values change -> interface element
                    self.InterElems[kint] = e
                    self.Elements[e].Dom = 0
                    kint += 1
                    break
                else:
                    if i+2 == self.n:   # if all nodal values hasve the same sign
                        if np.sign(LSe[i+1]) > 0:   # all nodal values with positive sign -> vacuum vessel element
                            self.VacuumElems[kvacuu] = e
                            self.Elements[e].Dom = +1
                            kvacuu += 1
                        else:   # all nodal values with negative sign -> plasma region element 
                            self.PlasmaElems[kplasm] = e
                            self.Elements[e].Dom = -1
                            kplasm += 1
                            
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.InterElems = self.InterElems[:kint]
        return
    
    
    def ComputeInterfaceApproximation(self):
        """ Compute the coordinates for the points describing the interface linear approximation. """
            
        for inter, elem in enumerate(self.InterElems):
            self.Elements[elem].InterfaceLinearApproximation()
            self.Elements[elem].interface = inter
        return
    
    def InitialiseVariables(self):
        self.PHI = np.zeros([self.Nn,self.itmax*self.itmax])  # All computed solutions matrix  
        self.PHI_inner0 = np.zeros([self.Nn])      # solution at inner iteration n
        self.PHI_inner1 = np.zeros([self.Nn])      # solution at inner iteration n+1
        self.PHI_outer0 = np.zeros([self.Nn])      # solution at outer iteration n
        self.PHI_outer1 = np.zeros([self.Nn])      # solution at outer iteration n+1
        self.PHI_converged = np.zeros([self.Nn])   # converged solution 
        return
    
    def ComputeIntegrationQuadratures(self):
        # COMPUTE STANDARD QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for elem in np.concatenate((self.PlasmaElems, self.VacuumElems), axis=0):
            self.Elements[elem].ComputeStandardQuadrature(self.QuadratureOrder)
            
        # COMPUTE MODIFIED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for elem in self.InterElems:
            self.Elements[elem].ComputeModifiedQuadrature(self.QuadratureOrder)
        return
    
    def ComputeInterfaceNormals(self):
        for elem in self.InterElems:
            self.Elements[elem].InterfaceNormal()
        self.CheckNormalVectors()
        return
    
    def CheckNormalVectors(self):
        for elem in self.InterElems:
            dir = np.array([self.Elements[elem].Xeint[1,0]-self.Elements[elem].Xeint[0,0], self.Elements[elem].Xeint[1,1]-self.Elements[elem].Xeint[0,1]]) 
            scalarprod = np.dot(dir,self.Elements[elem].NormalVec)
            if scalarprod > 1e-10: 
                raise Exception('Dot product equals',scalarprod, 'for mesh element', elem, ": Normal vector not perpendicular")
        return
    
    
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
                ELEMENT.PHI_Dg[ig] = self.AnalyticalSolution(ELEMENT.Xgint[ig,:])
                
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
    
    def CheckConvergence(self,loop):
        
        if loop == "INNER":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_inner1) > 0:
                L2residu = np.linalg.norm(self.PHI_inner1 - self.PHI_inner0)/np.linalg.norm(self.PHI_inner1)
            else: 
                L2residu = np.linalg.norm(self.PHI_inner1 - self.PHI_inner0)
            if L2residu < self.TOL_inner:
                self.marker_inner = False   # STOP WHILE LOOP 
                self.PHI_outer1 = self.PHI_inner1
            else:
                self.marker_inner = True
                self.PHI_inner0 = self.PHI_inner1
            
        elif loop == "OUTER":
            # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
            if np.linalg.norm(self.PHI_outer1) > 0:
                L2residu = np.linalg.norm(self.PHI_outer1 - self.PHI_outer0)/np.linalg.norm(self.PHI_outer1)
            else: 
                L2residu = np.linalg.norm(self.PHI_outer1 - self.PHI_outer0)
            if L2residu < self.TOL_outer:
                self.marker_outer = False   # STOP WHILE LOOP 
                self.PHI = self.PHI[:,:self.it+1]
                self.PHI_converged = self.PHI_outer1
            else:
                self.marker_outer = True
                self.PHI_outer0 = self.PHI_outer1
        return 
    
    def UpdateElementalPHI(self):
        for element in self.Elements:
            element.PHIe = self.PHI_inner1[element.Te]
        return

    
    def PlasmaEquilibrium(self):
        
        # INITIALISE VARIABLES
        self.InitialiseVariables()
        
        # INITIAL GUESS FOR MAGNETIC FLUX
        print("COMPUTE INITIAL GUESS...", end="")
        self.PHI_inner0 = self.InitialGuess()
        self.PHI_outer0 = self.PHI_inner0
        self.PHI[:,0] = self.PHI_inner0
        print('Done!')
        
        # INITIALISE LEVEL-SET FUNCTION
        print("INITIALISE LEVEL-SET...", end="")
        self.LevelSet = self.InitialLevelSet()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("INITIALISE ELEMENTS...", end="")
        self.Elements = [Element(e,self.ElType,self.ElOrder,self.X[self.T[e,:],:],self.T[e,:],self.LevelSet[self.T[e,:]],self.PHI_inner0[self.T[e,:]]) for e in range(self.Ne)]
        print('Done!')
        
        # CLASSIFY ELEMENTS  ->  OBTAIN PLASMAELEMS, VACUUMELEMS, INTERELEMS
        print("CLASSIFY ELEMENTS...", end="")
        self.ClassifyElements()
        print("Done!")
        
        # COMPUTE INTERFACE LINEAR APPROXIMATION
        print("APPROXIMATE INTERFACE...", end="")
        self.ComputeInterfaceApproximation()
        print("Done!")
        
        # COMPUTE INTERFACE APPROXIMATION NORMALS
        print('COMPUTE INTERFACE NORMALS...', end="")
        self.ComputeInterfaceNormals()
        print('Done!')
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        self.PlotSolution(self.PHI_inner0)
        
        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.marker_outer = True
        self.it_outer = 0
        self.it = 0
        while (self.marker_outer == True and self.it_outer < self.itmax):
            self.it_outer += 1
            self.marker_inner = True
            self.it_inner = 0
            while (self.marker_inner == True and self.it_inner < self.itmax):
                self.it_inner += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.it_outer)+' , INNER ITERATION = '+str(self.it_inner))
                self.AssembleGlobalSystem()
                #self.ApplyBoundaryConditions()
                self.SolveSystem()
                self.UpdateElementalPHI()
                self.PlotSolution(self.PHI_inner1)
                self.CheckConvergence("INNER")
                
            self.CheckConvergence("OUTER")
        
        self.PlotSolution(self.PHI_converged,colorbar=True)
            
        return
    
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
            PHIexact = self.AnalyticalSolution(self.X[i,:])
            error[i] = np.abs(PHIexact-phi[i])
            
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.tricontourf(self.X[:,0],self.X[:,1], error, levels=30)
        #plt.tricontour(self.X[:,0],self.X[:,1], PHIexact, levels=[0], colors='k')
        plt.colorbar()

        plt.show()
        
        return