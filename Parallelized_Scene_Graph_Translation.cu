/*
    CS 6023 Assignment 3.
    Do not make any changes to the boiler plate code or the other files in the folder.
    Use cudaFree to deallocate any memory not in usage.
    Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

void readFile(const char *fileName, std::vector<SceneNode *> &scenes, std::vector<std::vector<int>> &edges, std::vector<std::vector<int>> &translations, int &frameSizeX, int &frameSizeY)
{
    /* Function for parsing input file*/

    FILE *inputFile = NULL;
    // Read the file for input.
    if ((inputFile = fopen(fileName, "r")) == NULL)
    {
        printf("Failed at opening the file %s\n", fileName);
        return;
    }

    // Input the header information.
    int numMeshes;
    fscanf(inputFile, "%d", &numMeshes);
    fscanf(inputFile, "%d %d", &frameSizeX, &frameSizeY);

    // Input all meshes and store them inside a vector.
    int meshX, meshY;
    int globalPositionX, globalPositionY; // top left corner of the matrix.
    int opacity;
    int *currMesh;
    for (int i = 0; i < numMeshes; i++)
    {
        fscanf(inputFile, "%d %d", &meshX, &meshY);
        fscanf(inputFile, "%d %d", &globalPositionX, &globalPositionY);
        fscanf(inputFile, "%d", &opacity);
        currMesh = (int *)malloc(sizeof(int) * meshX * meshY);
        for (int j = 0; j < meshX; j++)
        {
            for (int k = 0; k < meshY; k++)
            {
                fscanf(inputFile, "%d", &currMesh[j * meshY + k]);
            }
        }
        // Create a Scene out of the mesh.
        SceneNode *scene = new SceneNode(i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity);
        scenes.push_back(scene);
    }

    // Input all relations and store them in edges.
    int relations;
    fscanf(inputFile, "%d", &relations);
    int u, v;
    for (int i = 0; i < relations; i++)
    {
        fscanf(inputFile, "%d %d", &u, &v);
        edges.push_back({u, v});
    }

    // Input all translations.
    int numTranslations;
    fscanf(inputFile, "%d", &numTranslations);
    std::vector<int> command(3, 0);
    for (int i = 0; i < numTranslations; i++)
    {
        fscanf(inputFile, "%d %d %d", &command[0], &command[1], &command[2]);
        translations.push_back(command);
    }
}

void writeFile(const char *outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY)
{
    /* Function for writing the final png into a file.*/
    FILE *outputFile = NULL;
    if ((outputFile = fopen(outputFileName, "w")) == NULL)
    {
        printf("Failed while opening output file\n");
    }

    for (int i = 0; i < frameSizeX; i++)
    {
        for (int j = 0; j < frameSizeY; j++)
        {
            fprintf(outputFile, "%d ", hFinalPng[i * frameSizeY + j]);
        }
        fprintf(outputFile, "\n");
    }
}

__global__ void doTranslations(int *translations, int *startX, int *startY, int *preorder, int *preOrderOffset, int *preOrderSize, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n)
    {
        return;
    }

    int mesh = translations[id * 3];
    int cmd = translations[id * 3 + 1];
    int amnt = translations[id * 3 + 2];

    // momentum in which we need to translate based on command
    int dirx[4] = {-1, 1, 0, 0};
    int diry[4] = {0, 0, -1, 1};

    int start = preOrderOffset[mesh];
    int sz = preOrderSize[mesh];

    for (int i = start; i < start + sz; i++)
    {
        atomicAdd(&startX[preorder[i]], dirx[cmd] * amnt);
        atomicAdd(&startY[preorder[i]], diry[cmd] * amnt);
    }
}

__global__ void generateScene(int **meshes, int *scene, int m, int n, int *startXs, int *startYs, int *opacity, int *frameSizeX, int *frameSizeY, int *opactiyMapToIndex, int V)
{
    long int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= m * n)
    {
        return;
    }

    // getting row and col of the scene we are going to fill
    int r = id / n;
    int c = id % n;

    if (scene[r * n + c] == -1)
    {
        // no mesh is overlapping here
        scene[r * n + c] = 0; // 0 as no mesh is present here
        return;
    }

    int meshInd = opactiyMapToIndex[scene[r * n + c]];

    int meshStartX = startXs[meshInd];
    int meshStartY = startYs[meshInd];

    int rel_x = r - meshStartX;
    int rel_y = c - meshStartY;

    scene[r * n + c] = meshes[meshInd][rel_x * frameSizeY[meshInd] + rel_y];
}

__global__ void genOpacityMatrix(int *scene, int m, int n, int *startXs, int *startYs, int *opacity, int *frameSizeX, int *frameSizeY, int V)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= V)
    {
        // out of bound
        return;
    }

    // we will iterate over all the points of the frame corresponding to v
    int startCorX = startXs[id];
    int startCorY = startYs[id];
    // mesh sizes
    int meshSizeX = frameSizeX[id];
    int meshSizeY = frameSizeY[id];

    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY; j++)
        {
            if (startCorX + i >= 0 && startCorX + i < m && startCorY + j >= 0 && startCorY + j < n)
            {
                // the index is going to lie on the final mesh
                atomicMax(&scene[(startCorX + i) * n + startCorY + j], opacity[id]);
                // to avoid race condition we are doing atomicmax
            }
        }
    }
}

__global__ void generateOpacityMap(int *opactiyMapToIndex, int *opacity, int V)
{

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= V)
    {
        // overflow
        return;
    }

    // we want to store opacity value to the index mapping
    int opacVal = opacity[id];
    // storing opacity value
    opactiyMapToIndex[opacVal] = id;
    // opactiyMapToIndex[val] -> will give the index of mesh corresponding to it
}

int preOrderTraversal(int *csr, int *CsrOffset, int node, int *index, int *preorder, int *preOrderSize, int *preOrderOffset)
{
    preorder[*index] = node;
    preOrderOffset[node] = *index;

    int start = CsrOffset[node];
    int end = CsrOffset[node + 1];

    // derefrencing and accessing the index
    *index = *index + 1;
    // incrementing the index

    int len = 1;
    // traversing over the child and getting sizes
    for (int i = start; i < end; i++)
    {
        len += preOrderTraversal(csr, CsrOffset, csr[i], index, preorder, preOrderSize, preOrderOffset);
    }

    preOrderSize[node] = len;
    // length of this subtree
    return len;
}

int main(int argc, char **argv)
{

    // Read the scenes into memory from File.
    const char *inputFileName = argv[1];
    int *hFinalPng;

    int frameSizeX, frameSizeY;
    std::vector<SceneNode *> scenes;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<int>> translations;
    readFile(inputFileName, scenes, edges, translations, frameSizeX, frameSizeY);
    hFinalPng = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);

    // Make the scene graph from the matrices.
    Renderer *scene = new Renderer(scenes, edges);

    // Basic information.
    int V = scenes.size();
    int E = edges.size();
    int numTranslations = translations.size();

    // Convert the scene graph into a csr.
    scene->make_csr(); // Returns the Compressed Sparse Row representation for the graph.
    int *hOffset = scene->get_h_offset();
    int *hCsr = scene->get_h_csr();
    int *hOpacity = scene->get_opacity();                      // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
    int **hMesh = scene->get_mesh_csr();                       // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
    int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX(); // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
    int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY(); // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
    int *hFrameSizeX = scene->getFrameSizeX();                 // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
    int *hFrameSizeY = scene->getFrameSizeY();                 // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

    auto start = std::chrono::high_resolution_clock::now();

    // Code begins here.
    // Do not change anything above this comment.

    // getting preorder data
    int *hpreorder;
    int *hpreorderoffset;
    int *hpreordersize;
    int *dpreorder;
    int *dpreorderoffset;
    int *dpreordersize;
    // preordersize[node] -> gives the size of the node value
    // preOrderOffset[node] -> gives the index where the node starts in preorder

    // host means cpu, device means gpu//

    hpreorder = (int *)malloc(sizeof(int) * (V + 1));
    hpreorderoffset = (int *)malloc(sizeof(int) * (V + 1));
    hpreordersize = (int *)malloc(sizeof(int) * (V + 1));

    cudaMalloc(&dpreorder, sizeof(int) * (V + 1));
    cudaMalloc(&dpreorderoffset, sizeof(int) * (V + 1));
    cudaMalloc(&dpreordersize, sizeof(int) * (V + 1));

    int ind = 0;
    // doing preorder traversal
    preOrderTraversal(hCsr, hOffset, 0, &ind, hpreorder, hpreordersize, hpreorderoffset);
    cudaMemcpy(dpreorder, hpreorder, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dpreorderoffset, hpreorderoffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dpreordersize, hpreordersize, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);

    // converting the 2d translations to 1d for to send to GPU
    int *hTranslations;
    int *dTranslations;
    hTranslations = (int *)malloc(sizeof(int) * numTranslations * 3);
    cudaMalloc(&dTranslations, sizeof(int) * numTranslations * 3);

    for (int i = 0; i < numTranslations; i++)
    {
        hTranslations[3 * i] = translations[i][0];
        hTranslations[3 * i + 1] = translations[i][1];
        hTranslations[3 * i + 2] = translations[i][2];
    }
    cudaMemcpy(dTranslations, hTranslations, sizeof(int) * numTranslations * 3, cudaMemcpyHostToDevice);

    // convert the hMesh to dMesh for that
    // need to do the deep copying
    int **dMesh;
    cudaMalloc(&dMesh, sizeof(int *) * V);
    int **tempMesh = (int **)malloc(sizeof(int *) * V);
    for (int i = 0; i < V; i++)
    {
        // create a copy of the mesh and store it
        int m = hFrameSizeX[i];
        int n = hFrameSizeY[i];
        int *mesh_address = hMesh[i];

        int *mesh_gpu;
        // allocating size of that mesh of ith index
        cudaMalloc(&mesh_gpu, m * n * sizeof(int));
        cudaMemcpy(mesh_gpu, mesh_address, m * n * sizeof(int), cudaMemcpyHostToDevice);
        tempMesh[i] = mesh_gpu;
    }

    cudaMemcpy(dMesh, tempMesh, sizeof(int *) * V, cudaMemcpyHostToDevice);

    // copying the adjaceny in the form of csr
    int *dOffSet;
    int *dCsr;
    int *dGlobalCoordinatesX;
    int *dGlobalCoordinatesY;

    cudaMalloc(&dOffSet, sizeof(int) * (V + 1));
    cudaMalloc(&dCsr, sizeof(int) * E);
    cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
    cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);

    cudaMemcpy(dOffSet, hOffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);
    cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);

    // doing translations based on the commands
    // launching threads per each instruction
    long int blocks = ceil(numTranslations / 1024.0);
    doTranslations<<<blocks, 1024>>>(dTranslations, dGlobalCoordinatesX, dGlobalCoordinatesY, dpreorder, dpreorderoffset, dpreordersize, numTranslations);

    int *DFinalPng;
    int *dOpacity;
    int *dFrameSizeX;
    int *dFrameSizeY;
    cudaMalloc(&DFinalPng, sizeof(int) * frameSizeX * frameSizeY);
    cudaMalloc(&dOpacity, sizeof(int) * V);
    cudaMalloc(&dFrameSizeX, sizeof(int) * V);
    cudaMalloc(&dFrameSizeY, sizeof(int) * V);

    // cudaMemcpy(DFinalPng, hFinalPng, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(dOpacity, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(dFrameSizeX, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(dFrameSizeY, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);

    // first generate the opacity matrix
    int opacityLimit = 300000009;
    int *opactiyMapToIndex;
    cudaMalloc(&opactiyMapToIndex, sizeof(int) * opacityLimit);
    cudaMemset(opactiyMapToIndex, -1, sizeof(int) * opacityLimit);
    blocks = ceil(V / 1024.0);
    // creating map for opacity value to the index
    generateOpacityMap<<<blocks, 1024>>>(opactiyMapToIndex, dOpacity, V);

    cudaMemset(DFinalPng, -1, sizeof(int) * frameSizeX * frameSizeY);
    blocks = ceil(V / 1024.0);
    genOpacityMatrix<<<blocks, 1024>>>(DFinalPng, frameSizeX, frameSizeY, dGlobalCoordinatesX, dGlobalCoordinatesY, dOpacity, dFrameSizeX, dFrameSizeY, V);
    blocks = ceil((frameSizeX * frameSizeY) / 1024.0);
    generateScene<<<blocks, 1024>>>(dMesh, DFinalPng, frameSizeX, frameSizeY, dGlobalCoordinatesX, dGlobalCoordinatesY, dOpacity, dFrameSizeX, dFrameSizeY, opactiyMapToIndex, V);

    // copying to the final image
    cudaMemcpy(hFinalPng, DFinalPng, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
    // freeing the memory
    cudaFree(DFinalPng);
    cudaFree(dOpacity);
    cudaFree(dFrameSizeX);
    cudaFree(dFrameSizeY);
    cudaFree(dOffSet);
    cudaFree(dCsr);
    cudaFree(dGlobalCoordinatesX);
    cudaFree(dGlobalCoordinatesY);
    cudaFree(dTranslations);
    cudaFree(dpreorder);
    cudaFree(dpreorderoffset);
    cudaFree(dpreordersize);

    // Do not change anything below this comment.
    // Code ends here.
    auto end = std::chrono::high_resolution_clock::now();
    // cudaDeviceSynchronize();

    std::chrono::duration<double, std::micro> timeTaken = end - start;
    printf("execution time : %f\n", timeTaken.count());
    // Write output matrix to file.
    const char *outputFileName = argv[2];
    writeFile(outputFileName, hFinalPng, frameSizeX, frameSizeY);
}
