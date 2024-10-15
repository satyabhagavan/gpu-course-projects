/**
 *   CS6023: GPU Programming
 *   Assignment 2
 *
 *   Please don't change any existing code in this file.
 *
 *   Please add necessary memory APIs for your implementation. Use cudaFree()
 *   to free up memory as soon as you're done with an allocation.
 *   This will ensure that you don't run out of memory while running
 *   large test cases. Use the minimum required memory for your
 *   implementation. DO NOT change the kernel configuration parameters.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;
#define TILE_WIDTH 32

__constant__ long int filter[51 * 51];
__global__ void convolutionShared(long int *d_mat, long int *d_ans, int m, int n, int k)
{
  // row from the y direction
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;

  // col from the x direction
  int tx = threadIdx.x;
  int col = blockIdx.x * blockDim.x + tx;

  __shared__ long int tile[TILE_WIDTH][TILE_WIDTH];

  // Assuming the filter is centered on the current element
  int filterRadius = k / 2;

  if (row >= m || col >= n)
  {
    return;
    // because it is going out of bound
  }

  long int sum = 0;
  // Load the current tile into shared memory
  if (tx < TILE_WIDTH && ty < TILE_WIDTH)
  {
    tile[ty][tx] = d_mat[row * n + col];
  }
  __syncthreads();
  // Wait for all threads to finish loading their part of the tile to write

  // getting the data to the shared memory

  // Apply the filter
  for (int i = -filterRadius; i <= filterRadius; i++)
  {
    for (int j = -filterRadius; j <= filterRadius; j++)
    {
      int currentRow = row + i;
      int currentCol = col + j;
      // getting this from the constant memory
      long int filterValue = filter[(filterRadius + i) * k + filterRadius + j];

      if (currentRow >= 0 && currentRow < m && currentCol >= 0 && currentCol < n)
      {
        long int matValue;
        // Check if the required element is within the shared memory tile
        int sharedRow = ty + i;
        int sharedCol = tx + j;
        if (sharedRow >= 0 && sharedRow < TILE_WIDTH && sharedCol >= 0 && sharedCol < TILE_WIDTH)
        {
          // Element is within the shared tile
          matValue = tile[sharedRow][sharedCol];
        }
        else
        {
          // Element is outside the shared tile, access it from global memory
          matValue = d_mat[currentRow * n + currentCol];
        }
        sum += matValue * filterValue;
      }
    }
  }

  // writng back to the answer
  if (row < m && col < n)
  {
    d_ans[row * n + col] = sum;
  }
}

int main(int argc, char **argv)
{

  int m, n, k;
  cin >> m >> n >> k;

  long int *h_mat = new long int[m * n];
  long int *h_filter = new long int[k * k];

  long int *h_ans = new long int[m * n];

  for (long int i = 0; i < m * n; i++)
  {
    cin >> h_mat[i];
  }

  for (long int i = 0; i < k * k; i++)
  {
    cin >> h_filter[i];
  }

  /**
   *
   * DO NOT CHANGE ANYTHING ABOVE THIS LINE
   *
   **/

  /****************************************************Start Here***********************************************************/

  long int *d_mat;
  long int *d_filter;
  long int *d_ans;

  cudaMalloc(&d_mat, m * n * sizeof(long int));
  cudaMalloc(&d_filter, k * k * sizeof(long int));
  cudaMalloc(&d_ans, m * n * sizeof(long int));

  cudaMemcpy(d_mat, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ans, h_ans, m * n * sizeof(long int), cudaMemcpyHostToDevice);

  // copying the filter to the constant memory
  cudaMemcpyToSymbol(filter, h_filter, k * k * sizeof(long int));

  long int noOfBlocks = ceil(m * n / 1024.0);

  auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

  dim3 blockSize(TILE_WIDTH, TILE_WIDTH); // Number of threads in each block

  // Calculating number of blocks along each dimension
  // Ceiling to ensure coverage of entire matrix even if it's not a multiple of TILE_WIDTH
  int numBlocksX = (n + TILE_WIDTH - 1) / TILE_WIDTH;
  int numBlocksY = (m + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 gridSize(numBlocksX, numBlocksY); // Number of blocks in the grid

  // Launch the kernel
  convolutionShared<<<gridSize, blockSize>>>(d_mat, d_ans, m, n, k);

  auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
  cudaDeviceSynchronize();

  cudaMemcpy(h_ans, d_ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);

  /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
  std::chrono::duration<double> elapsed1 = end - start;
  /**
   *
   * DO NOT CHANGE ANYTHING BELOW THIS LINE
   *
   */

  std::ofstream file("cuda.out");
  if (file.is_open())
  {
    for (long int i = 0; i < m; i++)
    {
      for (long int j = 0; j < n; j++)
      {
        file << h_ans[i * n + j] << " ";
      }
      file << "\n";
    }
    file.close();
  }
  else
  {
    std::cout << "Unable to open file";
  }

  std::ofstream file2("cuda_timing.out");
  if (file2.is_open())
  {
    file2 << elapsed1.count() << "\n";
    file2.close();
  }
  else
  {
    std::cout << "Unable to open file";
  }

  return 0;
}