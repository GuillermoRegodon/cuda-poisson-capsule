# cuda-poisson-capsule

CUDA capsule to check Poisson's equation solver that will be run in a Particle-In-Cell for plasma simulation of the interaction of a plasma with a metallic surface. It is an improvement over the code that has been used by Antonio Tejero-del-Caz in the work that he performed for his PhD. In his PIC code, most of the computation time was invested in solving Poisson's equation by means of the Jacobi iterative method, which has the advantage of being very simple to program, therefore facilitating the verification of the validity of the code, which was his main occupation. As part of the work for my PhD I have tested his code and separated the Poisson's equation solver into this capsule, in order to compare the methods.

Crank-Nicholson method has been proposed and tested as a improvement. This method involves two sums, of many elements, that cannot be performed in one CUDA block, therefore synchronization between blocks is an issue. The only way to synchronize blocks is to end the kernel and start another kernel (Computation Capability 1.1, in the system used for this work). The algorithm has been divided into five kernels. The algorithm is a modification of a scan algorithm with adapted coefficients to match the coefficients derived from theory.

It was also observed that the minimum number of iterations of the Jacobi method (which is twice the number of nodes) already gave a converged solution, so that in practice, every run of Jacobi method used the same number of iteration. Convergence test, by means of error checking is not necessary until the minimum number of iterations is reached, and given the computational cost of error checking, it is not recomended that is is checked every Jacobi iteration.

These are the conclusions (which are also written in the code as comments):

	// 1e6 repetitions of Crank-Nicholson method solver takes 108 seconds
  //        => each repetition takes 0.1ms (as expected)

	// 1000 repetitions of Jacobi method with error checking every iteration (original algorithm)
  // take 37 seconds
  //        => each repetition takes 37 ms

	// 10000 iteraciones de Jacobi method, modified for no error checking until necessary
	// take 42 segundos
  //        => cada una 4.2 ms, great improvement!!

  // CONCLUSION: Jacobi method without error checking is 9 times faster, which tells us that error checking should be performed only every 9 iterations or more
  //             Crank-Nicholson method is 370 times faster (programming time is worth the effort)

As a whole, Crank-Nicholson method for Poisson's equation solution is a great improvement. It makes the computation time of the Poisson's equation solver comparable to the computation time of the rest of the code.

This results where used, together with other improvements in the physics of the algorithm, in a contribution to the XXIVESCAMPIG International Conference in Glasgow:
 * "PIC simulation of a collisional planar pre-sheath", Regodón GF, Fernández Palop JI, Tejero-del-Caz A, Díaz-Cabrera JM, Carmona-Cabezas R and Ballesteros J
