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


from src.GaussQuadrature import *
from src.ShapeFunctions import *

class Coil:
    
    def __init__(self,index,dim,X,I):
        
        self.index = index      # COIL INDEX
        self.dim = dim          # SPATIAL DIMENSION
        self.X = X              # COIL POSITION COORDINATES
        self.I = I              # COIL CURRENT
        return
    
class Solenoid:
    
    def __init__(self,index,ElOrder,dim,Xe,I):
        
        self.index = index      # SOLENOID INDEX
        self.ElType = 0         # BAR (1D) ELEMENT
        self.ElOrder = ElOrder  # SOLENOID ELEMENTAL ORDER
        self.n = ElOrder + 1    # SOLENOID ELEMENT NUMBER OF NODES
        self.dim = dim          # SPATIAL DIMENSION
        self.Xe = Xe            # SOLENOID POSITION COORDINATES MATRIX
        self.I = I              # SOLENOID CURRENT
        
        # NUMERICAL INTEGRATION QUADRATURE
        self.ng = None          # NUMBER OF GAUSS INTEGRATION NODES FOR STANDARD 1D QUADRATURE
        self.XIg = None         # GAUSS INTEGRATION NODES (REFERENCE SPACE)
        self.Xg = None          # GAUSS INTEGRATION NODES (PHYSICAL SPACE)
        self.Wg = None          # GAUSS INTEGRATION WEIGTHS 
        self.Ng = None          # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None      # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None       # DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID
        return
    
    def ComputeHOnodes(self):
        XeHO = np.zeros([self.n,self.dim])
        XeHO[:2,:] = self.Xe
        dx = np.abs(self.Xe[1,0]-self.Xe[0,0])/(self.n-1)
        dy = np.abs(self.Xe[1,1]-self.Xe[0,1])/(self.n-1)
        for iinnernode in range(2,self.n):
            XeHO[iinnernode,:] = [self.Xe[0,0]+(iinnernode-1)*dx,
                                 self.Xe[0,1]+(iinnernode-1)*dy]
        self.Xe = XeHO
        return
    
    def ComputeIntegrationQuadrature(self,NumQuadOrder):
        
        # COMPUTE 1D NUMERICAL INTEGRATION QUADRATURES TO INTEGRATE ALONG SOLENOIDS
        self.XIg, self.Wg, self.ng = GaussQuadrature(self.ElType,NumQuadOrder)
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE 
        self.Ng, self.dNdxig, foo = EvaluateReferenceShapeFunctions(self.XIg, self.ElType, self.ElOrder)
        # MAP THE GAUSS INTEGRATION NODES TO PHYSICAL SPACE
        self.Xg = self.Ng@self.Xe
        # COMPUTE DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID 
        self.detJg = np.zeros([self.ng])
        for ig in range(self.ng):
            self.detJg[ig] = Jacobian1D(self.Xe,self.dNdxig[ig,:])
        return