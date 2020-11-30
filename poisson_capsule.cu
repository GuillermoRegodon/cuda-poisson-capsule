// Program to encapsulate Poisson Solver

#include <stdio.h>
#include <string>
#include <string.h>
#include <iostream>

using namespace std;

#define CST_ME 9.109e-31      // electron mass (kg)
#define CST_E 1.602e-19       // electron charge (C)
#define CST_KB 1.381e-23      // boltzmann constant (m^2 kg s^-2 K^-1)
#define CST_EPSILON 8.854e-12 // free space electric permittivity (s^2 C^2 m^-3 kg^-1)

#define CHARGE_DEP_BLOCK_DIM 512   //block dimension for particle2grid kernel
#define JACOBI_BLOCK_DIM 128       //block dimension for jacobi_iteration kernel
#define CN_SCAN_BLOCK_DIM 64       //block dimension for Crank-Nicolson scan kernel
#define CN_MAP_BLOCK_DIM 64       //block dimension for Crank-Nicolson map kernel

extern __shared__ double sh_mem[];

/******************************************************************************
 * Helper function to check cuda errors
******************************************************************************/

void cu_check(cudaError_t cuError, const string file, const int line)
{
  // function variables
  
  // function body
  
  if (0 == cuError)
  {
    return;
  } else
  {
    cout << "CUDA error found in file " << file << " at line " << line << ". (error code: " << cuError << ")" << endl;
    cout << "Exiting simulation" << endl;
    exit(1);
  }
  
}

/******************************************************************************
 * Helper function to synchronize the threads
******************************************************************************/

void cu_sync_check(const string file, const int line)
{
  // function variables
  cudaError_t cuError;
  
  // function body
  
  cudaDeviceSynchronize();
  cuError = cudaGetLastError();
  if (0 == cuError)
  {
    return;
  } else
  {
    cout << "CUDA error found in file " << file << " at line " << line << ". (error code: " << cuError << ")" << endl;
    cout << "Exiting simulation" << endl;
    exit(1);
  }
  
}

/******************************************************************************
 * Helper functions for the global static variables
******************************************************************************/

double init_ds(void) 
{
  // function variables
  static double ds = 0.0;
  
  // function body
  
  if (ds == 0.0) ds = 0.1;
  
  return ds;
}

/**********************************************************/

double init_Dl(void)
{
  // function variables
  static double Dl = 0.0;
  
  // function body
  
  if (Dl == 0.0) {
    double ne = 1.0e9;
    double Te = 1000.0;
    Dl = sqrt(CST_EPSILON*CST_KB*Te/(ne*CST_E*CST_E));
  }
  
  return Dl;
}

/**********************************************************/

double init_epsilon0(void) 
{
  // function variables
  double Te;
  const double Dl = init_Dl();
  static double epsilon0 = 0.0;
  // function body
  
  if (epsilon0 == 0.0) {
    Te = 1000.0;
    epsilon0 = CST_EPSILON;                         // SI units
    epsilon0 /= pow(Dl*sqrt(CST_ME/(CST_KB*Te)),2); // time units
    epsilon0 /= CST_E*CST_E;                        // charge units
    epsilon0 *= Dl*Dl*Dl;                           // length units
    epsilon0 *= CST_ME;                             // mass units
  }
  //Si epsilon es un double, tal y como están realizadas las operaciones, vale inf
  
  //return epsilon0;
  return 1;
  }

/**********************************************************/

int init_nn(void) 
{
  // function variables
  static int nn = 201;
  
  // function body
  
  return nn;
}










/******************************************************************************
 * 
 * Jacobi iteration of the Jacobi method, full version that calculates the 
 * maximum error of the solution, in order to test convergence of the method
 * 
******************************************************************************/

__global__ void jacobi_iteration (int nn, double ds, double epsilon0, double *g_rho, double *g_phi, double *g_error)
{
  /*----------------------------- function body -------------------------*/
  
  // shared memory pointers
  double *sh_old_phi= (double *) sh_mem;
  double *sh_error = (double *) &sh_old_phi[JACOBI_BLOCK_DIM+2];   // manually set up shared memory
  
  // registers
  double new_phi, dummy_rho;
  int tid = (int) threadIdx.x;
  int sh_tid = (int) threadIdx.x + 1;
  int g_tid = (int) (threadIdx.x + blockDim.x * blockIdx.x) + 1;
  int bdim = (int) blockDim.x;
  int bid = (int) blockIdx.x;
  int gdim = (int) gridDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global to shared memory
  if (g_tid < nn - 1) sh_old_phi[sh_tid] = g_phi[g_tid];

  // load comunication zones, load the edges of the tile of data
  if (bid < gdim-1) {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (sh_tid == bdim) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  } else {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (g_tid == nn-2) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  }
  
  // load charge density data into registers
  if (g_tid < nn - 1) {
  	dummy_rho = ds*ds*g_rho[g_tid]/epsilon0;
  }

  __syncthreads();
   
  // actualize interior mesh points
  if (g_tid < nn - 1) {
    new_phi = 0.5*(dummy_rho + sh_old_phi[sh_tid-1] + sh_old_phi[sh_tid+1]);
    
    // store new values of phi in global memory
    g_phi[g_tid] = new_phi;
    
    // evaluate local errors
    sh_error[tid] = fabs(new_phi-sh_old_phi[sh_tid]);
  }
  __syncthreads();

  // reduction for obtaining maximum error in current block
  for (int stride = 1; stride < bdim; stride <<= 1) {
    if ((tid%(stride*2) == 0) && (tid+stride < bdim) && (g_tid+stride < nn-1)) {
      if (sh_error[tid]<sh_error[tid+stride]) sh_error[tid] = sh_error[tid+stride];
    }
    __syncthreads();
  }
  
  // store maximun error in global memory
  if (tid == 0) g_error[bid] = sh_error[tid];
  
  return;
}










/******************************************************************************
 * 
 * Jacobi iteration of the Jacobi method, version that does not calculates
 * maximum error
 * 
******************************************************************************/

__global__ void jacobi_iter_no_error (int nn, double ds, double epsilon0, double *g_rho, double *g_phi)
{
  /*----------------------------- function body -------------------------*/
  
  // shared memory pointers
  double *sh_old_phi= (double *) sh_mem;
  
  // registers
  double new_phi, dummy_rho;
  int sh_tid = (int) threadIdx.x + 1;
  int g_tid = (int) (threadIdx.x + blockDim.x * blockIdx.x) + 1;
  int bdim = (int) blockDim.x;
  int bid = (int) blockIdx.x;
  int gdim = (int) gridDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global to shared memory
  if (g_tid < nn - 1) sh_old_phi[sh_tid] = g_phi[g_tid];

  // load comunication zones, load the edges of the tile of data
  if (bid < gdim-1) {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (sh_tid == bdim) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  } else {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (g_tid == nn-2) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  }
  
  // load charge density data into registers
  if (g_tid < nn - 1) {
  	dummy_rho = ds*ds*g_rho[g_tid]/epsilon0;
  }
  
  __syncthreads();

  // actualize interior mesh points
  if (g_tid < nn - 1) {
    new_phi = 0.5*(dummy_rho + sh_old_phi[sh_tid-1] + sh_old_phi[sh_tid+1]);
    
    // store new values of phi in global memory
    g_phi[g_tid] = new_phi;
    
  }
  
  return;
}










/******************************************************************************
 * 
 * This function solves Poisson's equation by means of the Jacobi method in the
 * GPU. Based in the work by Antonio Tejero-del-Caz for his PhD. Jacobi method
 * is iterative, and the error has to be calculated every iteration until it
 * reaches an acceptable value.
 * 
 * Checking error takes around 9 times longer than no checking, so we could 
 * check every 9 iterations or more.
 * 
 * In practice, as there is a minimum number of iterations according to theory,
 * the minimum number of iterations is always enough to obtain an acceptable
 * error. 
 * 
******************************************************************************/

void poisson_solver_jacobi(double max_error, double *d_rho, double *d_phi) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory pointers
  static const double ds = init_ds();               // spatial step
  static const int nn = init_nn();                  // number of nodes
  static const double epsilon0 = init_epsilon0();   // electric permitivity of free space
  
  double *h_error;
  double t_error = max_error*10;

  // Jacobi method is iterative, here we save the min number of iteration according
  // to theory so that the solution is valid. It turns out to be enough to have
  // the minimum error that it can be obtained using this method, so that it always
  // performs the same number of iterations
  int min_iteration = 2*nn;
  
  dim3 blockdim, griddim;
  size_t sh_mem_size;
  cudaError_t cuError;

  // device memory pointers
  double *d_error;
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim = JACOBI_BLOCK_DIM;
  griddim = (int) ((nn-2)/JACOBI_BLOCK_DIM) + 1;
  
  // define size of shared memory for jacobi_iteration kernel
  sh_mem_size = (2*JACOBI_BLOCK_DIM+2)*sizeof(double);
  
  // allocate host and device memory for vector of errors
  cuError = cudaMalloc((void **) &d_error, griddim.x*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  h_error = (double*) malloc(griddim.x*sizeof(double));
  
  int iter_count = 0;

  // execute jacobi iterations until solved
  // no need to check error until minimum number of interations is reached
  while(min_iteration>0) {
    // execute all but one iteration without checking errors
    cudaGetLastError();
    jacobi_iter_no_error<<<griddim, blockdim, sizeof(double)>>>(nn, ds, epsilon0, d_rho, d_phi);
    min_iteration--;
  }

  while (t_error>=max_error) {
    // execute at least one iteration checking errors
    cudaGetLastError();

    jacobi_iteration<<<griddim, blockdim, sh_mem_size>>>(nn, ds, epsilon0, d_rho, d_phi, d_error);
    cu_sync_check(__FILE__, __LINE__);
        
    // copy error vector from  device to host memory
    cuError = cudaMemcpy(h_error, d_error, griddim.x*sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
    
    // evaluate max error of the iteration
    t_error = 0;
    for (int i = 0; i<griddim.x; i++)
    {
      if (h_error[i] > t_error) t_error = h_error[i];
    }
    
    iter_count++;
  }

  printf("iter_count = %i\n", iter_count);

  // free device memory
  cudaFree(d_error);
  free(h_error);

  return;
}










/*************************************************************************
 * 
 * Here we find the modification to include the Crank-Nicholson method
 * for Poisson's equation solver. These kernels are executed in the same
 * order as they are declared.
 * 
 * The Crank-Nocholson algorithm is basicly a exact method that performs
 * two sums over the elements of the input array, with coefficients that
 * depend on the system of differental equations. As Poisson's equation
 * is the same in all the iterations, the method does not change. Moreover,
 * as the system has certain periodicities, the coefficients can be
 * calculated before programming time and introduced in the sums.
 * 
 * The sums over many elements are performed using the scan algorithm,
 * modified with the coefficients for the method. The synchronization
 * threads can be performed in the same kernel, but the synchronazation
 * between the blocks required ending the kernels and running the next.
 * 
*************************************************************************/

__global__ void cn_map_rho(double* d_temp1, double* d_rho, double ds, double epsilon0, int max_idx) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx<max_idx) {
		d_temp1[idx] = -((double) idx+1)*ds*ds*d_rho[idx]/epsilon0;
	}

}

/**********************************************************/

__global__ void cn_fw_part_scan(double *d_des, double *d_src, int max_idx) {
	// We start the algorithm knowing that we will need to use more than one block

	double *sh_src= (double *) sh_mem;
	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx<max_idx) {

		sh_src[threadIdx.x] = d_src[idx];

		__syncthreads();

		// SCAN OVER SH_src:
		// reduction over the elements on the left
		for (int stride = 1; stride<blockDim.x; stride = stride*2) {
			int k = threadIdx.x - stride;
			if (k>=0) {
				sh_src[threadIdx.x] += sh_src[k];
			}
			__syncthreads();
		}

		d_des[idx] = sh_src[threadIdx.x];
	}

	return;

}

/**********************************************************/

__global__ void cn_fw_glob_scan(double *d_des, double* d_src, int max_idx, double* d_phi) {
	// It is necessary to synchronize all the blocks, which can only be done by 
  // ending the kernel and starting a new one

	double *sh_acum = (double *) sh_mem;
	double *sh_phi_0 = (double *) &sh_mem[1];
	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (idx < max_idx) {

    // first thread saves the acum
		if (threadIdx.x == 0) {
			double r_acum = 0.0;
			for (int i = blockDim.x-1; i<blockIdx.x*blockDim.x; i+=blockDim.x) {
				r_acum += d_src[i];
			}
			sh_acum[0] = r_acum;
		}
    // first thread of the other blocks
		if ((threadIdx.x == blockDim.x-1)||(idx==max_idx-1)) {
			sh_phi_0[0] = d_phi[0];
		}
	}
	__syncthreads();
	if (idx < max_idx) {
		//d_des[idx] = sh_acum[0] + d_src[idx]; // this line woould be a simple scan, no coefs
		d_des[idx] = (sh_acum[0] + d_src[idx]- sh_phi_0[0])/((double) (idx+1)*(idx+2));
	}

}

/**********************************************************/

__global__ void cn_bw_part_scan(double *d_des, double *d_src, int max_idx) {
  // block kernel, d_des can be equal to d_src

	double *sh_src= (double *) sh_mem;
	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx<max_idx) {
		sh_src[threadIdx.x] = d_src[idx];

		__syncthreads();

		// SCAN over SH_src inverted:
		// reduction over the elements on the left
		for (int stride = 1; stride<blockDim.x; stride = stride*2) {
			int k = threadIdx.x + stride;
      // it is necessary to check if it is the last block
			if (blockIdx.x ==gridDim.x - 1) { 
				if (k<max_idx%blockDim.x) {
					sh_src[threadIdx.x] += sh_src[k];
				}
			} else {
				if (k<blockDim.x) {
					sh_src[threadIdx.x] += sh_src[k];
				}
			}
			__syncthreads();
		}

		d_des[idx] = sh_src[threadIdx.x];
	}

}	

/**********************************************************/

__global__ void cn_bw_glob_scan(double *d_des, double* d_src, int max_idx, double* d_phi_L) {
	// It is necessary to synchronize all the blocks, which can only be done by 
  // ending the kernel and starting a new one
	// global kernel, d_des cannot be the same as d_src

	double *sh_acum= (double *) sh_mem;
	double *sh_phi_L = (double *) &sh_mem[1];
	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx<max_idx) {

		if (threadIdx.x == 0) {
			double r_acum = 0.0;
			for (int i = (blockIdx.x+1)*blockDim.x; i<max_idx; i+=blockDim.x) {
				r_acum += d_src[i];
			}
			sh_acum[0] = r_acum;
		}
		if ((threadIdx.x == blockDim.x-1)||(idx==max_idx-1)) {
			sh_phi_L[0] = d_phi_L[0]/((double) max_idx+1); // max_idx+1 == nn-1
		}
		__syncthreads();
		d_des[idx] = (sh_phi_L[0]-sh_acum[0]-d_src[idx])*((double) idx+1);
	}
}

/**********************************************************/

void poisson_solver_cn(double max_error, double *d_rho, double *d_phi) {
	// It is necessary to synchronize all the blocks, which can only be done by 
  // ending the kernel and starting a new one

  // max_error is conserved for compatibility with poisson_solver_jacobi
  // In PIC code, just change the library, it has the same function poisson_solver
	
	// global variables in host
	int nn = init_nn();
	double epsilon0 = init_epsilon0();
	double ds = init_ds();

	cudaError_t cuError;

	// Allocate device memory
	double* d_temp1;
	cuError = cudaMalloc ((void **) &d_temp1, (nn-2)*sizeof(double));
	cu_check(cuError, __FILE__, __LINE__);
	double* d_temp2;
	cuError = cudaMalloc ((void **) &d_temp2, (nn-2)*sizeof(double));
	cu_check(cuError, __FILE__, __LINE__);

	// Size of grids for each part of the algorithm
	int map_blocks_per_grid = (nn+CN_MAP_BLOCK_DIM-1)/CN_MAP_BLOCK_DIM;
	int scan_blocks_per_grid = (nn-2+CN_SCAN_BLOCK_DIM-1)/CN_SCAN_BLOCK_DIM;

	// Shared mem for scan part
	size_t sh_mem_size;
	sh_mem_size = (CN_SCAN_BLOCK_DIM)*sizeof(double);

	// Obtain rho_1, stored in d_temp1
	cn_map_rho<<<map_blocks_per_grid, CN_MAP_BLOCK_DIM>>>(d_temp1, &d_rho[1], ds, epsilon0, nn-2);

	// Obtain D_2, stored in d_temp1. Modified to substract d_phi[0], even if it is not part of scan
	cn_fw_part_scan<<<scan_blocks_per_grid, CN_SCAN_BLOCK_DIM, sh_mem_size>>>(d_temp2, d_temp1, nn-2);
	cn_fw_glob_scan<<<scan_blocks_per_grid, CN_SCAN_BLOCK_DIM, 2*sizeof(double)>>>(d_temp1, d_temp2, nn-2, d_phi);

	// Obtain d_phi. Modified to add d_phi[nn-1]
	cn_bw_part_scan<<<scan_blocks_per_grid, CN_SCAN_BLOCK_DIM, sh_mem_size>>>(d_temp2, d_temp1, nn-2);
	cn_bw_glob_scan<<<scan_blocks_per_grid, CN_SCAN_BLOCK_DIM, 2*sizeof(double)>>>(&d_phi[1], d_temp2, nn-2, &d_phi[nn-1]);

	cudaFree(d_temp1);
	cudaFree(d_temp2);

}










/******************************************************************************
 * main
******************************************************************************/

int main(int argc, char** argv) {

	printf("Initiating Poisson Solver Capsule to compare different algorithms\n\n");

	cudaDeviceReset();

	int nn = init_nn();
	int nc = nn-1;
	double max_error = 1.0e-4;

	cudaError_t cuError;

	// Initialize the memories
	double* h_rho = (double*) malloc(nn*sizeof(double));
	double* h_phi = (double*) malloc(nn*sizeof(double));

	double* d_rho;
	double* d_phi;

	cuError = cudaMalloc ((void **) &d_rho, nn*sizeof(double));
	cu_check(cuError, __FILE__, __LINE__);
	cuError = cudaMalloc ((void **) &d_phi, nn*sizeof(double));
	cu_check(cuError, __FILE__, __LINE__);

	double phi_0 = 0.5;
	double phi_L = 2.0;
	double x_0 = 0.0;
	double ds = init_ds();
	double x_L = nn*ds;

	// Initialize h_rho anyway, no influence in the calculation time (Care with overflows)
 	for (int i=0; i<nn; i++) {
 		h_rho[i] = 1.0;
 		h_phi[i] = phi_0 + ((double) i)*(phi_L - phi_0)/((double) nc);
 	}

 	// Copy to device
 	cuError = cudaMemcpy(d_rho, h_rho, nn*sizeof(double), cudaMemcpyHostToDevice);
	cu_check(cuError, __FILE__, __LINE__);
	cuError = cudaMemcpy(d_phi, h_phi, nn*sizeof(double), cudaMemcpyHostToDevice);
	cu_check(cuError, __FILE__, __LINE__);

	// RUNS PoissonSolver once, uncomment to run
	poisson_solver_cn(max_error, d_rho, d_phi);

	// RUNS PoissonSolver many times, uncomment to run
	// for (int i=0;i<100000;i++) poisson_solver_cn(max_error, d_rho, d_phi);

	// RUNS PoissonSolver many times, uncomment to run
	// for (int i=0;i<1000;i++) poisson_solver_jacobi(max_error, d_rho, d_phi);





  // FROM WHERE we obtain the following measures
	// 1e6 repetitions of poisson_solver_cn takes 108 seconds
  //        => each repetition takes 0.1ms (as expected)

	// 1000 repetitions of jacobi with error calculation every iteration (original)
  // take 37 seconds
  //        => each repetition takes 37 ms

	// 10000 iteraciones de jacobi poisson_solver, modified for no error checking
	// take 42 segundos
  //        => cada una 4.2 ms, great improvement!!

  // CONCLUSION: Jacobi method without error checking is 9 times faster
  //             Crank-Nicholson method is 370 times faster (programming time is worth the effort)





	// Bring back d_phi to host
	cuError = cudaMemcpy(h_phi, d_phi, nn*sizeof(double), cudaMemcpyDeviceToHost);
	cu_check(cuError, __FILE__, __LINE__);
	cuError = cudaMemcpy(h_rho, d_rho, nn*sizeof(double), cudaMemcpyDeviceToHost);
	cu_check(cuError, __FILE__, __LINE__);

	for (int i=0; i<nn; i++) {
		printf("%g\t%g\t%g\n",x_0 + ((double) i)*(x_L-x_0)/((double) nc), h_rho[i], h_phi[i]);
	}

	// Free memory
	free(h_rho);
	free(h_phi);

	cudaFree (d_rho);
	cudaFree (d_phi);


}











