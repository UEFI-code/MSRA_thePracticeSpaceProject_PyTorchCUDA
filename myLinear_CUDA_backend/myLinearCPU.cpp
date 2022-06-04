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


#include <vector>

void myCell_forward_kernel_cpu(int i, int j, const float* input, const float* weight, float* output, const int Neuros, const int InputDim) 
{
	//Here InputDim == NumberOfSynapses
	const int CellID = j;
	const int BatchID = i;
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

void myKasoCell_backward_kernel_cpu(int i, int j, const float* input, const float* weight, float* output, const int KasoNeuros, const int InputDim)
{
	//Here InputDim == RealCellNumber, KasoNeuros == NumberOfSynapses
	const int KasoCellID = j;
	//KasoCellID match RealCell's pin
        const int BatchID = i;

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

std::vector<torch::Tensor> mylinear_cpu_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
    const int Batchsize = input.size(0);
    const int InputDim = input.size(1);
    const int Neuros = weights.size(0);

    auto output = torch::zeros({Batchsize, Neuros}, torch::TensorOptions().device(torch::kCUDA));

    //void *pCPUinput = 0, *pCPUweights = 0, *pCPUoutput = 0;

    //AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] { pCPUinput = input.data<scalar_t>(); pCPUweights = weights.data<scalar_t>(); pCPUoutput = output.data<scalar_t>(); }));
    
    float *pCPUinput = &(input.accessor<float,2>()[0][0]);
    float *pCPUweights = &(weights.accessor<float,2>()[0][0]);
    float *pCPUoutput = &(output.accessor<float,2>()[0][0]);

    printf("pCPUinput = 0x%x\n", pCPUinput);

    if(pCPUinput == 0)
	    exit(-1);

    for(int i = 0; i < Batchsize * InputDim; i++)
    	printf("%f\t", pCPUinput[i]);
    

    for(int i = 0; i < Batchsize; i++)
        for(int j = 0; j < Neuros; j++)
    	    myCell_forward_kernel_cpu(i, j, (float *)pCPUinput, (float *)pCPUweights, (float *)pCPUoutput, Neuros, InputDim);

    return {output};
}

std::vector<torch::Tensor> mylinear_cpu_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int Batchsize = grad_output.size(0);
    const int RealCellNum = grad_output.size(1);
    const int KasoCellNum = weights.size(1);

    auto grad_input = torch::zeros({Batchsize, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));
    auto grad_weights = torch::zeros({RealCellNum, KasoCellNum}, torch::TensorOptions().device(torch::kCUDA));

    void *pCPUgrad_input, *pCPUgrad_weights, *pCPUgrad_output, *pCPUinput, *pCPUweights;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] {
			    pCPUgrad_input = grad_input.data<scalar_t>();
			    pCPUgrad_weights = grad_weights.data<scalar_t>();
			    pCPUgrad_output = grad_output.data<scalar_t>();
			    pCPUinput = input.data<scalar_t>();
			    pCPUweights = weights.data<scalar_t>();
			    }));
    
    for(int i = 0; i < Batchsize; i++)
	for(int j = 0; j < KasoCellNum; j++)
	    myKasoCell_backward_kernel_cpu(i, j, (float *)pCPUgrad_output, (float *)pCPUweights, (float *)pCPUgrad_input, KasoCellNum, RealCellNum);

    return {grad_input, grad_weights};
}