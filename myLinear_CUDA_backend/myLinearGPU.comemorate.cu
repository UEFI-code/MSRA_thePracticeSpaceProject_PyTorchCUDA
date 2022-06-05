// MIT License

// Copyright (c) Microsoft Corporation and SuperHacker UEFI.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void myCell_forward_kernel(const float* input, const float* weight, float* output, const int Neuros, const int InputDim) 
{
	//Here InputDim == NumberOfSynapses
	const int CellID = threadIdx.x;
	const int BatchID = blockIdx.x;
	const float *myWeightBase = weight + CellID * InputDim;
	const float *myInputBase = input + BatchID * InputDim;
	float *myOutput = output + BatchID * Neuros + CellID;
	
	*myOutput = 0.0;

	for(int i=0; i<InputDim; i++)
	{
		*myOutput += myWeightBase[i] * myInputBase[i];
	}

	return;
}

__global__ void myKasoCell_backward_kernel(const float* input, const float* weight, float* output, const int KasoNeuros, const int InputDim)
{
	//Here InputDim == RealCellNumber, KasoNeuros == NumberOfSynapses
	const int KasoCellID = threadIdx.x;
	//KasoCellID match RealCell's pin
        const int BatchID = blockIdx.x;

	const float *myInput = input + BatchID * InputDim;
	const float *myWeight = weight + KasoCellID;
	float *myOutput = output + KasoCellID + BatchID * KasoNeuros;
	*myOutput = 0.0;

	for(int i = 0; i < InputDim; i++)
	{
		*myOutput += myWeight[i * KasoNeuros] * myInput[i];
	}

	return;
}

template <typename scalar_t>
__global__ void ms_demo_matmul_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M, 
    const int K, 
    const int N,
    const bool trans_A = false,
    const bool trans_B = false) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        scalar_t sum = 0.0;
        for (int k = 0; k < K; k++)
        {
            const int i = trans_A ? (k * M + row) : (row * K + k);
            const int j = trans_B ? (col * K + k) : (k * N + col);
            sum += A[i] * B[j];
        }

        C[row * N + col]  = sum;
    }
}

std::vector<torch::Tensor> mylinear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
    printf("Here is CPU\n");
    const int Batchsize = input.size(0);
    const int InputDim = input.size(1);
    const int Neuros = weights.size(0);

    auto output = torch::zeros({Batchsize, Neuros}, torch::TensorOptions().device(torch::kCUDA));

    void *pGPUinput = 0, *pGPUweights = 0, *pGPUoutput = 0;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] { pGPUinput = input.data<scalar_t>(); pGPUweights = weights.data<scalar_t>(); pGPUoutput = output.data<scalar_t>(); }));

    float *pCPUinput = (float *)malloc(sizeof(float) * M * K);
    float *pCPUweights = (float *)malloc(sizeof(float) * K * N);
    cudaMemcpy((void *)pCPUinput, (void *)pGPUinput, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)pCPUweights, (void *)pGPUweights, sizeof(float) * K * N, cudaMemcpyDeviceToHost);
    
    for(int i=0; i < 30; i++)
	    printf("input.data[%d] = %f\tweights.data[%d] = %f\n", i, pCPUinput[i], i, pCPUweights[i]);

    myCell_forward_kernel<<<Batchsize, Neuros>>>((float *)pGPUinput, (float *)pGPUweights, (float *)pGPUoutput, Neuros, InputDim);

    return {output};
}

std::vector<torch::Tensor> mylinear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int Batchsize = grad_output.size(0);
    const int RealCellNum = grad_output.size(1);
    const int KasoCellNum = weights.size(1);

    auto grad_input = torch::zeros({Batchsize, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));
    auto grad_weights = torch::zeros({RealCellNum, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(32, 32);
    const dim3 grid1((Batchsize - 1) / 32 + 1, (KasoCellNum - 1) / 32 + 1);
    const dim3 grid2((RealCellNum - 1) / 32 + 1, (KasoCellNum - 1) / 32 + 1);

    void *pGPUgrad_input, *pGPUgrad_weights, *pGPUgrad_output, *pGPUinput, *pGPUweights;

    //printf("input.size(0) = %d, input.size(1) = %d\n", input.size(0), input.size(1));
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] {
			    pGPUgrad_input = grad_input.data<scalar_t>();
			    pGPUgrad_weights = grad_weights.data<scalar_t>();
			    pGPUgrad_output = grad_output.data<scalar_t>();
			    pGPUinput = input.data<scalar_t>();
			    pGPUweights = weights.data<scalar_t>();
			    }));
    
    float *pCPUinput = (float *)malloc(sizeof(float) * input.size(0) * input.size(1));
    float *pCPUweights = (float *)malloc(sizeof(float) * weights.size(0) * weights.size(1));

    cudaMemcpy((void *)pCPUinput, (void *)pGPUinput, sizeof(float) * input.size(0) * input.size(1), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)pCPUweights, (void *)pGPUweights, sizeof(float) * weights.size(0) * weights.size(1), cudaMemcpyDeviceToHost);

    printf("\n----Dump input----\n");
    
    for(int i=0; i < input.size(0) * input.size(1); i++)
	    printf("%f\t", pCPUinput[i]);

    printf("\n----Dump weights----\n");

    for(int i=0; i < weights.size(0) * weights.size(1); i++)
            printf("%f\t", pCPUweights[i]);	    

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        ms_demo_matmul_kernel<scalar_t><<<grid1, block>>>(
            grad_output.data<scalar_t>(),
            weights.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            M,
            N,
            K,
            false,
            false);
        }));

    myKasoCell_backward_kernel<<<Batchsize, KasoCellNum>>>((float *)pGPUgrad_output, (float *)pGPUweights, (float *)pGPUgrad_input, KasoCellNum, RealCellNum);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        ms_demo_matmul_kernel<scalar_t><<<grid2, block>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            grad_weights.data<scalar_t>(),
            RealCellNum,
            Batchsize,
            KasoCellNum,
            true,
            false);

        }));

    float *pCPUgrad_weights = (float *)malloc(sizeof(float) * RealCellNum * KasoCellNum);

    cudaMemcpy((void *)pCPUgrad_weights, (void *)pGPUgrad_weights, sizeof(float) * N * K, cudaMemcpyDeviceToHost);

    printf("\n----Dump grad_weights----\n");

    for(int i=0; i < N * K; i++)
            printf("%f\t", pCPUgrad_weights[i]);

    printf("\n");

    return {grad_input, grad_weights};
}
