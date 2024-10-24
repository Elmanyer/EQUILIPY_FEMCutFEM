# **PYTHON GRAD-SHAFRANOV SOLVER BASED ON CUTFEM**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system. Solving the equation yields the plasma equilibrium cross-section configuration.

The problem is tackle as a free-boundary problem, where the plasma cross-section geometry is free to evolve and deform towards the equilibrium state. In order to deal with such configuration, the solver is based on a CutFEM numerical scheme, a non-conforming mesh Finite Element Method where the geometry is embedded in the mesh. 

The input files *.equ.dat* and meshes available in the repository have been designed according to the ITER tokamak cross-section geometry. In order to launch simulations with other geometries, the user shall change the geometrical parameters for the vacuum vessel and the external coils and solenoids from the input file *.equ.dat*.

## *CONTENT:*
- folder **src**: contains the source code
- folder **CASES**: contains the input files *.equ.dat* for the different problem cases
- folder **MESHES**: contains the files describing different meshes
- folder **TESTs**: contains the test-suites, in both *.py* and *.ipynb* format, for the different standard problem cases, and the main test files in *.ipynb*

## *EXECUTION:*

After clonning the repository, the code is ready is run. 

Inside the **TESTs** folder, the user may find the test-suites *TS-* files, both in *.py* and *.ipynb*, ready to execute. The mesh used for the simulation may be changed by commenting and uncommenting the adequate lines. These test-suites represent the simulations corresponding to the *FIXED*-boundary analytical cases, for the *LINEAR*, *NONLINEAR* and *ZHENG* plasma current models, and the *FREE*-boundary problem with *PROFILES* plasma current model.

To launch other simulations, the user may use testing files *MainTestEquilipyFIXED.ipynb* and *MainTestEquilipyFREE.ipynb*, where all available meshes have been included. For fixed-boundary problem simulations, meshes can be adjusted to the fixed plasma cross-section (*-REDUCED* meshes); on the other hand, for free-boundary problem simulations larger meshes should be used, preparing for plasma cross-ection deformations. 

