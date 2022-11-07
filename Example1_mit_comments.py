


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   ## to clearfiy and clean the multi-kernel environemt.



###This is a first example of how to use lettuce. A two dimensional Taylor Green vortex is initialized and simulated for 10000 steps. Afterwards the energy and the velocity field is plotted.


import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch


##METHOD SETUP
##Reporters will grab the results in between simulation steps (see reporters.py and simulation.py)
##Output: Column 1: simulation steps, Column 2: time in LU, Column 3: kinetic energy in PU
##Output: separate VTK-file with ux,uy,(uz) and p for every 100. time step in ./output



lattice = lt.Lattice(lt.D2Q9, device = "cpu", dtype=torch.float32) ##Lattice energy can be defined as the energy required to convert one mole of an ionic solid into gaseous ionic constituents.
flow = lt.TaylorGreenVortex2D(resolution=256, reynolds_number=100, mach_number=0.05, lattice=lattice) ##Taylor–Green vortex is an unsteady flow of a decaying vortex, which has an exact closed form solution of the incompressible Navier–Stokes equations in Cartesian coordinates. 
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu) ## 'BGK' a collision operator used in the Boltzmann equation and in the lattice Boltzmann method, a computational fluid dynamics technique.
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

Energy = lt.IncompressibleKineticEnergy(lattice, flow) ##relating the energy with the flow
reporter = lt.ObservableReporter(Energy, interval=1000, out=None) ##setting intervales
simulation.reporters.append(reporter) ##A reporter that prints an observable every few iterations.
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=100, filename_base="./output"))

simulation.initialize_f_neq() ##to approximate the contributions by finite differences
mlups = simulation.step(num_steps=10000) ##to convert string or numbers to a floating point number
print("Performance in MLUPS:", mlups)

##Energy Reporter

energy = np.array(simulation.reporters[0].out) ##printing the plot
print(energy.shape)
plt.plot(energy[:,1],energy[:,2])
plt.title('Kinetic energy')
plt.xlabel('Time')
plt.ylabel('Energy in physical units')
plt.show()
(11, 3)

##Velocity
u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).numpy()
u_norm = np.linalg.norm(u,axis=0)
plt.imshow(u_norm)
plt.show()