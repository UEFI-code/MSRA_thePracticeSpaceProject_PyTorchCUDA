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
#include <stdlib.h>
#include <vector>

void myCell_forward_kernel_cpu(int BatchID, int CellID, const float* input, const float* weight, const float* Kakuritsu, float* output, const int Neuros, const int InputDim, const unsigned int timeNow) 
{
	//Here InputDim == NumberOfSynapses
	const float *myWeightBase = weight + CellID * InputDim;
	const float *myKakuriBase = Kakuritsu + CellID * InputDim;
	const float *myInputBase = input + BatchID * InputDim;
	float *myOutput = output + BatchID * Neuros + CellID;
	
	*myOutput = 0.0;
	
	srand(timeNow + BatchID + CellID);
	float RandNum = 0.0;

	for(int i=0; i<InputDim; i++)
	{
		RandNum = (rand() % 4096) / 4096.0;
		if(RandNum < myKakuriBase[i])
		    *myOutput += myWeightBase[i] * myInputBase[i];
		//printf("myOutput = %f\n", *myOutput);
	}

	return;
}

void myKasoCell_backward_kernel_cpu(int BatchID, int KasoCellID, const float* input, const float* weight, float* output, const int KasoNeuros, const int InputDim)
{
	//Here InputDim == RealCellNumber, KasoNeuros == NumberOfSynapses
	//KasoCellID match RealCell's pin

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

std::vector<torch::Tensor> myKakuritsu_cpu_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor Kakuritsu)
{
    const int Batchsize = input.size(0);
    const int InputDim = input.size(1);
    const int Neuros = weights.size(0);

    auto output = torch::zeros({Batchsize, Neuros}, torch::TensorOptions());

    float *pCPUinput = input.data_ptr<float>();
    float *pCPUweights = weights.data_ptr<float>();
    float *pCPUKakuritsu = Kakuritsu.data_ptr<float>();
    float *pCPUoutput = output.data_ptr<float>();

    /*
    printf("pCPUinput = 0x%x\n", pCPUinput);
    
    for(int i = 0; i < Batchsize * InputDim; i++)
    	printf("%f\t", pCPUinput[i]);

    printf("\n");
    */
    //while(1); //Wait for debug

    for(int i = 0; i < Batchsize; i++)
        for(int j = 0; j < Neuros; j++)
    	    myCell_forward_kernel_cpu(i, j, pCPUinput, pCPUweights, pCPUKakuritsu, pCPUoutput, Neuros, InputDim, (unsigned int)time(NULL));

    return {output};
}

std::vector<torch::Tensor> myKakuritsu_cpu_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int Batchsize = grad_output.size(0);
    const int RealCellNum = grad_output.size(1);
    const int KasoCellNum = weights.size(1);

    auto grad_input = torch::zeros({Batchsize, KasoCellNum}, torch::TensorOptions());
    auto grad_weights = torch::zeros({RealCellNum, KasoCellNum}, torch::TensorOptions());

    float *pCPUgrad_input = grad_input.data<float>();
    float *pCPUgrad_weights = grad_weights.data<float>();
    float *pCPUgrad_output = grad_output.data<float>();
    float *pCPUinput = input.data<float>();
    float *pCPUweights = weights.data<float>();

    for(int i = 0; i < Batchsize; i++)
	for(int j = 0; j < KasoCellNum; j++)
	    myKasoCell_backward_kernel_cpu(i, j, pCPUgrad_output, pCPUweights, pCPUgrad_input, KasoCellNum, RealCellNum);

    return {grad_input, grad_weights};
}
