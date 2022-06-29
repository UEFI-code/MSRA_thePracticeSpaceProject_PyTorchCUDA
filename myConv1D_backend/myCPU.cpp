// MIT License

// Copyright (c) Microsoft Corporation and Cookie.

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

void myCell_forward_kernel_cpu(const int BaID, const int KeID, const float* Input, const float* weight, float* Output, const int TuchuNUM, const int Expectshape, const int KeNUM, const int Inputshape, const int ChNum, const int Stride) 
{
	//Here InputDim == NumberOfSynapses
	const float *WBase = weight + KeID * TuchuNUM;
	const float *InBase = Input + BaID * Inputshape * ChNum;
	float *InBasetemp = 0;
	float *OutBase = Output + BaID * Expectshape * KeNUM + KeID * Expectshape;
	int TargetCh = 0, MicroBias = 0;
	for(int i=0; i < Expectshape; i++)
	{
		InBasetemp = (float *)InBase + i * Stride;
		for(int j=0;j<TuchuNUM; j++)
		{
			TargetCh = j / ChNum;
			MicroBias = j % ChNum;
			OutBase[i] += InBasetemp[TargetCh * Inputshape + MicroBias] * WBase[j];
		}
	}

	return;
}

void myCell_backward_kernel_cpu(const int BaID, const int KeID, const float* Input, const float* weight, float* Output, const int TuchuNUM, const int KeNUM, const int Expectshape, const int Inputshape, const int ChNum, const int Stride)
{
         const float *WBase = weight + KeID * TuchuNUM;
	 const float *InBase = Input + BaID * Inputshape * KeNUM + KeID * Inputshape;
	 float *OutBase = Output + BaID * Expectshape * ChNum;
	 float *OutBasetemp = 0;
	 int TargetCh = 0, MicroBias = 0;
         for(int i=0; i < Inputshape; i++)
	 {
		 OutBasetemp = (float *)OutBase + i * Stride;
		 for(int j=0;j<TuchuNUM; j++)
		 {
			 TargetCh = j / ChNum;
			 MicroBias = j % ChNum;
			 OutBasetemp[TargetCh * Inputshape + MicroBias] += InBase[i] * WBase[j];
		 }
	 }
	 return;
}

std::vector<torch::Tensor> myConv1d_cpu_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int Stride)
{
    const int Batchsize = input.size(0);
    const int ChNum = input.size(1);
    const int inputShape = input.size(2);

    const int KeNum = weights.size(0);
    const int KeSize = weights.size(2);

    const int ExpectShape = ((inputShape - KeSize) / Stride) + 1;

    auto output = torch::zeros({Batchsize, KeNum, ExpectShape}, torch::TensorOptions());

    const float *pCPUinput = input.data<float>();
    const float *pCPUweights = weights.data<float>();
    float *pCPUoutput = output.data<float>();
    for(int i =0; i < Batchsize; i++)
	    for(int j=0; j < KeNum; j++)
		    myCell_forward_kernel_cpu(i, j, pCPUinput, pCPUweights, pCPUoutput, ChNum * KeSize, ExpectShape, KeNum, inputShape, ChNum, Stride);

    return {output};
}

std::vector<torch::Tensor> myConv1d_cpu_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    int Stride)
{
    const int Batchsize = grad_output.size(0);
    const int inputShape = grad_output.size(2);
    const int KeNum = weights.size(0);
    const int OutChNum = weights.size(1);
    const int KeSize = weights.size(2);
    const int ExpectShape = (inputShape - 1) * Stride + KeSize;

    auto grad_input = torch::zeros({Batchsize, OutChNum, ExpectShape}, torch::TensorOptions());
    auto grad_weights = torch::zeros({KeNum, OutChNum, KeSize}, torch::TensorOptions());

    const float *pCPUinput = grad_output.data<float>();
    const float *pCPUweights = weights.data<float>();
    float *pCPUoutput = grad_input.data<float>();
    for(int i =0; i < Batchsize; i++)
            for(int j=0; j < KeNum; j++)
		    myCell_backward_kernel_cpu(i, j, pCPUinput, pCPUweights, pCPUoutput, OutChNum * KeSize, KeNum, ExpectShape, inputShape, OutChNum, Stride);

    return {grad_input, grad_weights};
}
