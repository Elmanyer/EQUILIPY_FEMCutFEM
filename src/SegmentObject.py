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


# This script contains the definition for class SEGMENT, an object which 
# embodies the segments contituting the mesh. 


class Segment:
    
    def __init__(self,index,ElOrder,Xeint,int_edges):
        
        self.int_index = index      # GLOBAL INDEX OF INTERFACE
        self.ElOrder = ElOrder      # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.dim = len(Xeint[0,:])  # SPATIAL DIMENSION
        self.nedge = ElOrder+1      # NUMBER OF NODES ON SEGMENT (1D ELEMENT)
        self.Xeint = Xeint          # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.int_edges = int_edges  # ELEMENTAL EDGES (VERTICES LOCAL INDEXES) ON WHICH ARE LOCATED SEGMENT ENDS
        self.Teint = None           # BOUNDARY EDGES CONNECTIVITY MATRIX (LOCAL INDEXES) 
        self.PSI_g = None           # PSI VALUE CONSTRAINT ON INTERFACE EDGES INTEGRATION POINTS
        
        # MODIFIED QUADRATURE FOR INTEGRATION ALONG INTERFACE 
        self.Ngaussint = None    # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 1D QUADRATURE
        self.Wgint = None        # GAUSS INTEGRATION WEIGHTS IN STANDARD 1D QUADRATURE
        self.XIgint = None       # MODIFIED GAUSS INTEGRATION NODES COMPUTED FROM 1D STANDARD QUADRATURE 
        self.Nint = None         # REFERENCE 2D SHAPE FUNCTIONS EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.Xgint = None        # PHYSICAL 2D GAUSS INTEGRATION NODES MAPPED FROM 1D REFERENCE ELEMENT
        self.dNdxiint = None     # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES 
        self.dNdetaint = None    # REFERENCE 2D SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT MODIFIED 1D GAUSS INTEGRATION NODES
        self.detJgint = None     # MATRIX DETERMINANTS OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT INTERFACE EVALUATED AT GAUSS INTEGRATION NODES
        
        # NORMAL VECTOR (OUTWARDS) 
        self.NormalVec = None       # INTERFACE/BOUNDARY EDGE NORMAL VECTOR POINTING OUTWARDS
        
        return