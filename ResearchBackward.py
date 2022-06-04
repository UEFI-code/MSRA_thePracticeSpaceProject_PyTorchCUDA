# BSD 3-Clause License

# Copyright (c) 2022, SuperHacker UEFI 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file has been changed for education and teaching purpose

import torch
import torch.nn as nn

#import our cuda module
#import mylinear_cpp
import myLinear_cuda

class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = myLinear_cuda.forward(input, weight)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #print(grad_output)
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        #grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        grad_input, grad_weight = myLinear_cuda.backward(grad_output, input, weight)

        print(grad_input)

        return grad_input, grad_weight

class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)

Neurons = 5
SynapseEachNeurons = 10
BatchSize = 4

layer = myLinear(10, 5).cuda()

x1 = torch.ones(1, SynapseEachNeurons)
x2 = torch.ones(1, SynapseEachNeurons) * 2
x3 = torch.ones(1, SynapseEachNeurons) * 3
x4 = torch.ones(1, SynapseEachNeurons) * 4

x = torch.cat([x1, x2, x3, x4], 0).cuda()

y = layer(x)

print(layer.weight)
print(x)
print(y)

L1Losser = nn.L1Loss()

target = torch.ones(BatchSize, Neurons).cuda()

loss = L1Losser(y, target)

loss.backward()
