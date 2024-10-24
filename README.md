# **PYTHON GRAD-SHAFRANOV SOLVER BASED ON CUTFEM**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system. Solving the equation yields the plasma equilibrium cross-section configuration.

The problem is tackle as a free-boundary problem, where the plasma cross-section geometry is free to evolve and deform towards the equilibrium state. In order to deal with such configuration, the solver is based on a CutFEM numerical scheme, a non-conforming mesh Finite Element Method where the geometry is embedded in the mesh. 

## *CONTENT:*
- folder **src**: contains the source code
- folder **CASES**: contains the input files for the different problem cases
- folder **MESHES**: contains the files describing different meshes
- folder **TESTs**: contains the test-suites, in both .py and .ipynb format, for the different standard problem cases, and the main test files in .ipynb

##*EXECUTION:*

After clonning the repository, the code is ready is run. 


