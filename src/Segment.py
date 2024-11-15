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

class InterfaceApprox:
    
    def __init__(self,index,Nsegments):
        
        self.index = index            # GLOBAL INDEX OF INTERFACE APPROXIMATION
        self.Nsegments = Nsegments    # NUMBER OF SEGMENTS CONSTITUTING INTERFACE APPROXIMATION
        self.Segments = None          # SEGMENT OBJECTS CONSTITUTING THE INTERFACE APPROXIMATION
        self.Xint = None              # INTERFACE APPROXIMATION NODAL COORDINATES MATRIX (PHYSICAL SPACE)
        self.XIint = None             # INTERFACE APPROXIMATION NODAL COORDINATES MATRIX (REFERENCE SPACE)
        self.Tint = None              # INTERFACE APPROXIMATION SEGMENTS CONNECTIVITY 
        self.ElIntNodes = None        # ELEMENTAL VERTICES INDEXES ON EDGES CUTING THE INTERFACE 
        return

class Segment:
    
    def __init__(self,index,ElOrder,Xseg):
        
        self.index = index          # SEGMENT INDEX
        self.ElType = 0             # ELEMENT TYPE = 0 ->> BAR ELEMENT
        self.ElOrder = ElOrder      # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.dim = len(Xseg[0,:])   # SPATIAL DIMENSION
        self.n = ElOrder+1          # NUMBER OF NODES ON SEGMENT (1D ELEMENT)
        self.Xseg = Xseg            # ELEMENTAL NODAL COORDINATES MATRIX (PHYSICAL SPACE)
        self.XIseg = None           # ELEMENTAL NODAL COORDINATES MATRIX (REFERENCE SPACE) 
        self.PSIgseg = None         # PSI VALUE ON SEGMENT GAUSS INTEGRATION NODES
        
        # QUADRATURE FOR INTEGRATION ALONG INTERFACE 
        self.ng = None              # NUMBER OF GAUSS INTEGRATION NODES 
        self.Wg = None              # GAUSS INTEGRATION WEIGHTS 
        self.XIg = None             # GAUSS INTEGRATION NODAL COORDINATES (REFERENCE SPACE)
        self.Xg = None              # GAUSS INTEGRATION NODAL COORDINATES (PHYSICAL SPACE)
        self.Ng = None              # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None          # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdetag = None         # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANTS OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL 
        
        self.NormalVec = None       # SEGMENT NORMAL VECTOR POINTING OUTWARDS (RESPECT TO INTERFACE)
        return
    