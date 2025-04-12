
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/cuda/CUDAEvent.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define DYNAMIC_GAUSS //Comment if you want to use static parameters in tiled convolution
#define DYNAMIC_SOBEL //Comment if you want to use static parameters in tiled convolution

#define SOBEL_MASK_SIZE 3
#define SOBEL_TILE_WIDTH 30
#define SOBEL_BLOCK_WIDTH (SOBEL_TILE_WIDTH + SOBEL_MASK_SIZE - 1)

int defaultMaskSize = 3;
float gxDefault[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
float gyDefault[3][3] = {{-1, 0, 1}, {-2,0,2}, {-1,0,1 } };

__device__ int getIndex(int x, int y, int width){
	return x * width + y;
}

float* createGaussianKernel( int size , double sigma ){
	float* kernel = new float[size * size];
	double mean = size/2;
	double sum = 0.0;

	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel[x * size + y] = exp( -0.5 * ( ( x - mean ) * ( x - mean ) + ( y - mean ) * ( y - mean ) )/(sigma * sigma) ); //why 0.5?
            sum += kernel[x * size + y];
		}
	}
 
	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel[x * size + y] /= sum;
		}
	}

	return kernel;
}

float* getGradientX(){
	float* data = new float[9];
	for (int i = 0; i < defaultMaskSize; ++i){
		for (int j = 0; j < defaultMaskSize; ++j){
			data[i * defaultMaskSize + j] = gxDefault[i][j];
		}
	}
	return data;
}

float* getGradientY(){
	float* data = new float[9];
	for (int i = 0; i < defaultMaskSize; ++i){
		for (int j = 0; j < defaultMaskSize; ++j){
			data[i * defaultMaskSize + j] = gyDefault[i][j];
		}
	}
	return data;
}


float* allocateArray(int size){
	float* data = new float[size];
	return data;
}

void freeArray(float* data){
	delete[] data;
}

__global__ void DynamicTiledConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ mask, int maskSize , int TILE_WIDTH, int BLOCK_WIDTH){
	extern __shared__ float imageDS[]; //Dynamic shared memory
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * TILE_WIDTH + ty;
	int colIn = blockIdx.x * TILE_WIDTH + tx;
	int radio = maskSize / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;
	
	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[getIndex(ty, tx, BLOCK_WIDTH )] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[getIndex(ty, tx, BLOCK_WIDTH)] = 0.0;

	__syncthreads();

	//Convolve image with gaussian mask
	float sum = 0.0;
	if (ty < TILE_WIDTH && tx < TILE_WIDTH){
		for (int i = 0; i < maskSize; ++i){
			for (int j = 0; j < maskSize; ++j){
				sum += mask[getIndex(i, j, maskSize)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sum;
	}
}

__global__ void VecOperation(const float* A, const float* B, float* C, int N, int OP){


	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N){
		if(OP==0) // Addition(+)
		{
			C[i] = A[i] + B[i];
		
		}else if(OP==1){ // Substract(-)
			
			C[i] = A[i] - B[i];
		
		}else if(OP==2){ // Multiplication(*)
			
			C[i] = A[i] * B[i];
	
		}else if(OP==3){ // Division(/)
		
			C[i] = A[i] / (B[i] + 1E-8);
		
		}
	
	
	}
	__syncthreads();

}

__global__ void DynamicSobelConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ gx, const float* __restrict__ gy, int TILE_WIDTH, int BLOCK_WIDTH, int dim){
	extern __shared__ float imageDS[]; //Dynamic shared memory
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * TILE_WIDTH + ty;
	int colIn = blockIdx.x * TILE_WIDTH + tx;
	int radio = SOBEL_MASK_SIZE / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;
	
	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[getIndex(ty, tx, BLOCK_WIDTH )] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[getIndex(ty, tx, BLOCK_WIDTH)] = 0.0;

	__syncthreads();

	//Convolve image with Gradient Masks
	float sumG = 0.0;
	if (ty < TILE_WIDTH && tx < TILE_WIDTH){
		for (int i = 0; i < SOBEL_MASK_SIZE; ++i){
			for (int j = 0; j < SOBEL_MASK_SIZE; ++j){
				if(dim==0){
					sumG += gx[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
				}else if(dim==1){
					sumG += gy[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
				}
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sumG;
	}
}

void parallelHarrisCornerDetectorCudaLauncher(float* grayImageArray, float* RHost, int imageWidth, int imageHeight){
	int kernelSize = 7;
	int imageSize = imageWidth * imageHeight;
	
	int numThreads = 32;
	int threadsPerBlock = 128;
	int blocksPerGrid = (imageSize + threadsPerBlock - 1)/threadsPerBlock;
	cudaError_t err;

	//Arrays of CPU
	//float* result = allocateArray(imageSize);
	float* IxHost = allocateArray(imageSize);
	float* IyHost = allocateArray(imageSize);
	float* SxxHost = allocateArray(imageSize);
	float* SyyHost = allocateArray(imageSize);
	float* SxyHost = allocateArray(imageSize);
	//float* RHost = allocateArray(imageSize);
	float* gaussKernelArray = createGaussianKernel(7, 1.5);
	float* xGradient = getGradientX(), *yGradient = getGradientY(); //Used in Sobel

	//Arrays used in GPU
	float* kernelDevice; //Used in Gauss
	float* xGradDevice, *yGradDevice; //Used in Sobel
	float* imageDevice, *resultDevice; //Used in both Gauss and Sobel
	float* IxDevice, *IyDevice; //Used in both Gauss and Sobel
	float* IxxDevice, *IyyDevice, *IxyDevice; //Used in both Gauss and Sobel
	float* SxxDevice, *SyyDevice, *SxyDevice; //
	float* tmp1Device, *tmp2Device, *detMDevice, *trMDevice, *ResponseDevice; //

	//Allocate Device Memory
	cudaMalloc((void**)&imageDevice, imageSize * sizeof(float));
	cudaMalloc((void**)&resultDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IyyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SxxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SyyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SxyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&tmp1Device, imageSize* sizeof(float));
	cudaMalloc((void**)&tmp2Device, imageSize* sizeof(float));
	cudaMalloc((void**)&detMDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&trMDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&ResponseDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&kernelDevice, kernelSize * kernelSize * sizeof(float));
	cudaMalloc((void**)&xGradDevice, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float));
	cudaMalloc((void**)&yGradDevice, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float));

	//Copy values from CPU to GPU
	cudaMemcpy(imageDevice, grayImageArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelDevice, gaussKernelArray, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(xGradDevice, xGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yGradDevice, yGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	
	/*
		 Gaussian Blur to Remove Noise
	*/
	#ifdef DYNAMIC_GAUSS
		int tileWidth = numThreads - kernelSize + 1;
		int blockWidth = tileWidth + kernelSize - 1;
		dim3 dimGaussBlock(blockWidth, blockWidth);
		dim3 dimGaussGrid((imageWidth - 1) / tileWidth + 1, (imageHeight - 1) / tileWidth + 1);
	#else
		dim3 dimGaussBlock(GAUSS_BLOCK_WIDTH, GAUSS_BLOCK_WIDTH);
		dim3 dimGaussGrid((imageWidth - 1) / GAUSS_TILE_WIDTH + 1, (imageHeight - 1) / GAUSS_TILE_WIDTH + 1);
	#endif
	
	//Start recording GPU time
	#ifdef DYNAMIC_GAUSS
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
	#else	
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
	#endif	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();

	
	/*
		 Sobel Filter to Compute Gx and Gy	
	*/
	//Tiled Sobel Filter
	#ifdef DYNAMIC_SOBEL
		int sobelTileWidth = numThreads - SOBEL_MASK_SIZE + 1;
		int sobelBlockWidth = sobelTileWidth + SOBEL_MASK_SIZE - 1;
		dim3 dimBlockSobel(sobelBlockWidth, sobelBlockWidth);
		dim3 dimGridSobel((imageWidth - 1) / sobelTileWidth + 1, (imageHeight - 1) / sobelTileWidth + 1);
	#else
		dim3 dimBlockSobel(SOBEL_BLOCK_WIDTH, SOBEL_BLOCK_WIDTH);
		dim3 dimGridSobel((imageWidth - 1) / SOBEL_TILE_WIDTH + 1, (imageHeight - 1) / SOBEL_TILE_WIDTH + 1);
	#endif
	
	// Copy result obtained from Gauss as new Image Data
	cudaMemcpy(imageDevice, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToDevice);
	
	#ifdef DYNAMIC_SOBEL
		DynamicSobelConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, IxDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, sobelTileWidth, sobelBlockWidth, 0);
		DynamicSobelConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, IyDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, sobelTileWidth, sobelBlockWidth, 1);
	#else	
		tiledSobelConvolution << < dimGridSobel, dimBlockSobel >> > (imageDevice, IxDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, 0);
		tiledSobelConvolution << < dimGridSobel, dimBlockSobel >> > (imageDevice, IyDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, 1);
	#endif

	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Sobel Error: %s\n", cudaGetErrorString(err));

	/*
		Calculate Ixx, Iyy and Ixy
	*/
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IxDevice, IxDevice, IxxDevice, imageSize, 2); // 2: Multiplication(*)
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IyDevice, IyDevice, IyyDevice, imageSize, 2); // 2: Multiplication(*)
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IxDevice, IyDevice, IxyDevice, imageSize, 2); // 2: Multiplication(*)
	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("VecOpeartion 0 Error: %s\n", cudaGetErrorString(err));

	/*
	 	Caluculate Sxx, Syy and Sxy with Gaussian Filter
	 */
	#ifdef DYNAMIC_GAUSS
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxxDevice, SxxDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IyyDevice, SyyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxyDevice, SxyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
	#else
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IxxDevice, SxxDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IyyDevice, SyyDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IxyDevice, SxyDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
	#endif	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();


	/* Compute R = detM / trM*/
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, tmp1Device, imageSize, 2); // 2: Multiplication
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxyDevice, SxyDevice, tmp2Device, imageSize, 2); // 
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(tmp1Device, tmp2Device, detMDevice, imageSize, 1); // 0: Addition, detM = Sxx * Syy - Sxy * Sxy

	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, trMDevice, imageSize, 0); // 0: Addition, trM = Sxx + Syy
	
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(detMDevice, trMDevice, ResponseDevice, imageSize, 3); // 3: devision
	
	//VecOpResponse<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, SxyDevice, ResponseDevice, imageSize);
	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("VecOperation Error: %s\n", cudaGetErrorString(err));

	// GPU output
	cudaMemcpy(RHost, ResponseDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free Memory used in GPU and CPU
	cudaFree(xGradDevice);	cudaFree(yGradDevice); cudaFree(imageDevice);
	cudaFree(resultDevice); cudaFree(kernelDevice); 
	cudaFree(IxxDevice);cudaFree(IyyDevice);cudaFree(IxyDevice);
	cudaFree(SxxDevice);cudaFree(SyyDevice);cudaFree(SxyDevice);
	freeArray(xGradient);	freeArray(yGradient);  //freeArray(result);
	freeArray(gaussKernelArray);
}