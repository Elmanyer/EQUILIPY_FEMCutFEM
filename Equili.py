""" This script contains the Python object defining a plasma equilibrium problem, modeled using the Grad-Shafranov equation
in an axisymmetrical system such as a tokamak. """

import numpy as np
import matplotlib.pyplot as plt
from random import random
from GaussQuadrature import *
from ShapeFunctions import *
from PlasmaCurrent import *
from ElementObject import *

class Equili:
    
    # GENERAL PARAMETERS
    epsilon0 = 8.8542E-12       # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    K = 1.602E-19               # J eV-1   Botlzmann constant

    def __init__(self,folder_loc,ElementType,ElementOrder,Rmax,Rmin,epsilon,kappa,delta):
        self.directory = folder_loc
        self.case = folder_loc[folder_loc.rfind("/")+1:]
        
        # DECLARE ATTRIBUTES
        self.ElType = ElementType
        self.ElOrder = ElementOrder
        self.epsilon = epsilon
        self.kappa = kappa
        self.delta = delta
        self.Rmax = Rmax
        self.Rmin = Rmin
        self.R0 = (Rmax+Rmin)/2
        self.QuadratureOrder = 2
        self.TOL_inner = 1e-3
        self.TOL_outer = 1e-3
        self.itmax = 5
        self.beta = 1e5  # NITSCHE'S METHOD PENALTY TERM

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
                    GRAD-SHAFRANOV EQ:  DELTA*(PHI) = R^2
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
        return coeffs 
    
    def InitialGuess(self):
        """ Use the analytical solution for the LINEAR case as initial guess. The plasma region is characterised by a negative phi in this solution. """
        # ADIMENSIONALISE MESH
        Xstar = self.X/self.R0
        phi0 = np.zeros([self.Nn])
        self.coeffs = self.ComputeLinearSolutionCoefficients()
        for i in range(self.Nn):
            phi0[i] = Xstar[i,0]**4/8 + self.coeffs[0] + self.coeffs[1]*Xstar[i,0]**2 + self.coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
            phi0[i] *= 2*random()
        return phi0
    
    
    def InitialLevelSet(self):
        """ Use the analytical solution for the LINEAR case as initial Level-Set function. The plasma region is characterised by a negative value of Level-Set. """
        # ADIMENSIONALISE MESH
        Xstar = self.X/self.R0
        LS0 = np.zeros([self.Nn])
        self.coeffs = self.ComputeLinearSolutionCoefficients()
        for i in range(self.Nn):
            LS0[i] = Xstar[i,0]**4/8 + self.coeffs[0] + self.coeffs[1]*Xstar[i,0]**2 + self.coeffs[2]*(Xstar[i,0]**4-4*Xstar[i,0]**2*Xstar[i,1]**2)
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
        self.PHI_inner0 = np.zeros([self.Nn])      # solution at inner iteration n
        self.PHI_inner1 = np.zeros([self.Nn])      # solution at inner iteration n+1
        self.PHI_outer0 = np.zeros([self.Nn])      # solution at outer iteration n
        self.PHI_outer1 = np.zeros([self.Nn])      # solution at outer iteration n+1
        self.PHI_converged = np.zeros([self.Nn])   # solution at outer iteration n+1
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
                    SourceTermg[ig] = self.mu0*ELEMENT.Xg2D[ig,0]*Jphi(self.mu0,ELEMENT.Xg2D[ig,0],ELEMENT.Xg2D[ig,1],PHIg[ig]) 
            # COMPUTE ELEMENTAL MATRICES
            LHSe, RHSe = ELEMENT.IntegrateElementalDomainMatrices(SourceTermg)
            # ASSEMBLE INTO GLOBAL SYSTEM
            for i in range(ELEMENT.n):
                for j in range(ELEMENT.n):
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] = LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] = RHSe[i]
                
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
                        SourceTermg[ig] = self.mu0*SUBELEM.Xg2D[ig,0]*Jphi(self.mu0,SUBELEM.Xg2D[ig,0],SUBELEM.Xg2D[ig,1],PHIg[ig])
                
                # COMPUTE ELEMENTAL MATRICES
                LHSe, RHSe = SUBELEM.IntegrateElementalDomainMatrices(SourceTermg)
                # ASSEMBLE INTO GLOBAL SYSTEM
                for i in range(SUBELEM.n):
                    for j in range(SUBELEM.n):
                        self.LHS[SUBELEM.Te[i],SUBELEM.Te[j]] = LHSe[i,j]
                    self.RHS[SUBELEM.Te[i]] = RHSe[i]    
                     
            ####### COMPUTE INTERFACE TERMS
            # COMPUTE INTERFACE CONDITIONS PHI_D
            PHI_Dg = np.zeros([ELEMENT.Ng1D])
            for ig in range(ELEMENT.Ng1D):
                PHI_Dg[ig] = self.BoundaryCondition(ELEMENT.Xgint[ig,:])
                
            # COMPUTE ELEMENTAL MATRICES
            LHSe, RHSe = ELEMENT.IntegrateElementalInterfaceMatrices(PHI_Dg,self.beta)
            # ASSEMBLE INTO GLOBAL SYSTEM
            for i in range(ELEMENT.n):
                for j in range(ELEMENT.n):
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] = LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] = RHSe[i]
            
        print("Done!")
        
        return
    
    
    def BoundaryCondition(self, x):
        
        # ADIMENSIONALISE 
        xstar = x/self.R0
        phiD = xstar[0]**4/8 + self.coeffs[0] + self.coeffs[1]*xstar[0]**2 + self.coeffs[2]*(xstar[0]**4-4*xstar[0]**2*xstar[1]**2)
        
        return phiD
    
    
    def SolveSystem(self):
        
        self.PHI_inner1 = np.linalg.solve(self.LHS, self.RHS)
        
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
        while (self.marker_outer == True and self.it_outer < self.itmax):
            self.it_outer += 1
            self.marker_inner = True
            self.it_inner = 0
            while (self.marker_inner == True and self.it_inner < self.itmax):
                self.it_inner += 1
                self.AssembleGlobalSystem()
                #self.ApplyBoundaryConditions()
                self.SolveSystem()
                print('OUTER ITERATION = '+str(self.it_outer)+' , INNER ITERATION = '+str(self.it_inner))
                self.UpdateElementalPHI()
                self.PlotSolution(self.PHI_inner1)
                
                self.CheckConvergence("INNER")
                
                
            self.CheckConvergence("OUTER")
            
        return
    
    def PlotSolution(self,phi):
        if len(np.shape(phi)) == 2:
            phi = phi[:,0]
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.tricontourf(self.X[:,0],self.X[:,1], phi, levels=30)
        plt.tricontour(self.X[:,0],self.X[:,1], phi, levels=[0], colors='k')
        #plt.colorbar()

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