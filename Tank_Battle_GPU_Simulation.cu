/*
    Roll No: CS23Mo65
    Name: Satya Bhagavan
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <limits.h>

using namespace std;

//*******************************************

// Write down the kernels here
__global__ void initialise(int *score, int *dReadHP, int *dWriteHP, int H, int T, int *ActiveCount, int *k)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= T)
    {
        return;
    }
    if (id == 0)
    {
        *k = 1;
        *ActiveCount = T;
    }
    score[id] = 0;
    dReadHP[id] = H;
    dWriteHP[id] = H;
}

__global__ void simulateFight(int *dXCoord, int *dYCoord, int *dReadHP, int *dWriteHP, int *dscore, int *k, int T)
{
    int source = blockIdx.x;
    int target = threadIdx.x;

    __shared__ long long int minDistance;
    __shared__ long long int finalTarget;

    if (threadIdx.x == 0)
    {
        minDistance = LLONG_MAX;
        finalTarget = -1;
    }
    __syncthreads();

    int target_dir_ind = (source + *k) % T;
    // corresponding to the direction
    long long int diffx = dXCoord[target_dir_ind] - dXCoord[source];
    long long int diffy = dYCoord[target_dir_ind] - dYCoord[source];

    // corresponding to the target
    long long int diff_x = dXCoord[target] - dXCoord[source];
    long long int diff_y = dYCoord[target] - dYCoord[source];
    long long int presentDistance = abs(diff_x) + abs(diff_y);

    if (dReadHP[target] <= 0)
    {
        // target is dead won't be able to contribute
    }
    else if (target != source)
    {
        // checking for the possible attackings

        if (diffx * diff_y == diffy * diff_x && diff_x * diffx >= 0 && diffy * diff_y >= 0 && dReadHP[target] > 0)
        {
            // lies on the same line and same quadrant
            atomicMin(&minDistance, presentDistance);
        }
    }

    __syncthreads();
    if (minDistance == LLONG_MAX)
    {
        // no attack option found we don't do anythin
    }
    else if (diffx * diff_y == diffy * diff_x && diff_x * diffx >= 0 && diffy * diff_y >= 0 && dReadHP[source] > 0 && dReadHP[target] > 0 && minDistance == presentDistance)
    {
        // this is the target and only one will be present such that
        // updating the scores
        // printf("%d, %d -> %d\n", *k, source, finalTarget);
        finalTarget = target;
        atomicAdd(&dscore[source], 1);
        atomicAdd(&dWriteHP[finalTarget], -1);
    }
}

__global__ void checkActive(int *dReadHP, int *dWriteHP, int *ActiveCount, int *k, int T)
{
    __shared__ int count;
    if (threadIdx.x == 0)
    {
        // initilisation
        count = 0;
    }
    __syncthreads();
    int id = threadIdx.x;
    // updating the readHP with the write
    dReadHP[id] = dWriteHP[id];

    if (dReadHP[id] > 0)
    {
        atomicAdd(&count, 1);
    }

    __syncthreads();
    if (id == 0)
    {
        *ActiveCount = count;

        // increasing the round
        if ((*k + 1) % T == 0)
        {
            *k = 1;
        }
        else
        {
            *k = *k + 1;
        }
    }
}

//***********************************************

int main(int argc, char **argv)
{
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *dscore;
    int *dXCoord;
    int *dYCoord;
    int *dReadHP;
    int *dWriteHP;
    int *ActiveCount;
    int *k;

    // allocating memory
    cudaMalloc(&dscore, sizeof(int) * T);
    cudaMalloc(&dXCoord, sizeof(int) * T);
    cudaMalloc(&dYCoord, sizeof(int) * T);
    cudaMalloc(&dReadHP, sizeof(int) * T);
    cudaMalloc(&dWriteHP, sizeof(int) * T);
    // host alloc memory for having two way communication
    cudaHostAlloc((void **)&ActiveCount, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&k, sizeof(int), cudaHostAllocMapped);

    cudaMemcpy(dXCoord, xcoord, sizeof(int) * T, cudaMemcpyHostToDevice);
    cudaMemcpy(dYCoord, ycoord, sizeof(int) * T, cudaMemcpyHostToDevice);

    // initilising the values
    initialise<<<1, T>>>(dscore, dReadHP, dWriteHP, H, T, ActiveCount, k);
    cudaDeviceSynchronize();

    while (*ActiveCount > 1)
    {
        // launch a simulation
        simulateFight<<<T, T>>>(dXCoord, dYCoord, dReadHP, dWriteHP, dscore, k, T);
        // for checking active counts and increasing the k
        checkActive<<<1, T>>>(dReadHP, dWriteHP, ActiveCount, k, T);
        cudaDeviceSynchronize();
        // printf("Active Count: %d\n", *ActiveCount);
    }

    // copying the final scores
    cudaMemcpy(score, dscore, sizeof(int) * T, cudaMemcpyDeviceToHost);

    // freeing up the memory
    cudaFree(dscore);
    cudaFree(dReadHP);
    cudaFree(dWriteHP);
    cudaFree(dXCoord);
    cudaFree(dYCoord);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}