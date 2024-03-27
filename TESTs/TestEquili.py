
from GradShafranovCutFEM import *

# LOCATION OF PROBLEM FOLDER
folder_loc = '/home/elmanyer/Documents/BSC/MasterThesis/Code/execution/TS-'
case = 'UNSTRUCTURED'
case = 'UNSTRUCTURED_COARSE'
case = 'UNSTRUCTURED_COARSE-FINE'
case = 'UNSTRUCTURED_ULTRACOARSE'
case = 'UNSTRUCTURED_ULTRAFINE'

# MESH INFORMATION:
ElementType = 1     # Element type -> 1: TRIANGLE  ; 2: QUADRILATERAL
ElementOrder = 1    # Element order -> 1: LINEAR  ; 2: QUADRATIC ; 3: CUBIC ...

# VACUUM VESSEL GEOMETRY (ITER):
epsilon = 0.32            # inverse aspect ratio
kappa = 1.7               # elongation
delta = 0.33              # triangularity
Rmax = 8                  # plasma major radius
Rmin = 4                  # plasma minor radius

##############################################################

# DECLARE OBJECT PROBLEM (folder with ALYA files)
directory = folder_loc + case
Problem = GradShafranovCutFEM(directory,ElementType,ElementOrder,Rmax,Rmin,epsilon,kappa,delta)

Problem.ReadMesh()
Problem.PlasmaEquilibrium() 





fig, axs = plt.subplots(1, 3, figsize=(18,8))

self = Problem
phi = self.PHI_converged
if len(np.shape(phi)) == 2:
    phi = phi[:,0]
axs[0].set_ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
axs[0].set_xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
axs[0].tricontourf(self.X[:,0],self.X[:,1], phi, levels=30)
axs[0].tricontour(self.X[:,0],self.X[:,1], phi, levels=[0], colors='k')

for elem in self.PlasmaBoundElems:
    ELEMENT = self.Elements[elem]

    # PLOT PHYSICAL INTERFACE LINEAR APPROXIMATION
    for i in range(ELEMENT.n):
        axs[0].plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], 
                [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='red', linewidth=1)


    # PLOT PHYSICAL INTERFACE LINEAR APPROXIMATION
    for i in range(ELEMENT.n):
        axs[1].plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], 
                [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=3)

    axs[1].plot(ELEMENT.Xeint[:,0], ELEMENT.Xeint[:,1], '.', color='red',markersize=10)
    axs[1].plot(ELEMENT.Xeint[:,0], ELEMENT.Xeint[:,1],color='red', linewidth=3)
    # PLOT NODE NUMERATION
    d = 0.005
    for i in range(len(ELEMENT.Xemod[:,0])):
        axs[1].text(ELEMENT.Xemod[i,0]+d,ELEMENT.Xemod[i,1]+d,str(i),fontsize=15)

    # PLOT TESSELLATION ON PHYSICAL ELEMENT
    colorlist = ['blue', 'red', 'green']
    markerlist = ['^', '<', '>']
    for i, subelem in enumerate(ELEMENT.SubElements):
        for j in range(subelem.n):   # plot edges
            axs[2].plot([subelem.Xe[j,0], subelem.Xe[int((j+1)%subelem.n),0]], [subelem.Xe[j,1], subelem.Xe[int((j+1)%subelem.n),1]],
                    linestyle = '-', color = colorlist[i], linewidth=2 , marker=markerlist[i])
    # PLOT MODIFIED PHYSICAL GAUSS INTEGRATION NODES
    for i, subelem in enumerate(ELEMENT.SubElements):
        axs[2].plot(subelem.Xg2D[:,0],subelem.Xg2D[:,1],'x',color=colorlist[i], markersize = 8)
        
    axs[1].set_title('Element '+str(elem))
    plt.pause(0.01)
    input()
    axs[1].cla()
    axs[2].cla()
    
    
plt.show()