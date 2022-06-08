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
import myLinear_benchmark
import os

Neurons = 50
SynapseEachNeurons = 10
BatchSize = 4

epoch = 100

UseCUDA = False

weight = nn.Parameter(torch.randn(Neurons, SynapseEachNeurons))

x1 = torch.ones(1, SynapseEachNeurons)
x2 = torch.ones(1, SynapseEachNeurons) * 2
x3 = torch.ones(1, SynapseEachNeurons) * 3
x4 = torch.ones(1, SynapseEachNeurons) * 4

x = torch.cat([x1, x2, x3, x4], 0)

if UseCUDA:
    weight = weight.cuda()
    x = x.cuda()
    try:
        os.remove('/tmp/myLinear_CUDA.txt')
    except:
        pass
else:
    try:
        os.remove('/tmp/myLinear_CPU.txt')
    except:
        pass

for i in range(epoch):
    y = myLinear_benchmark.forward(x, weight)[0]

if UseCUDA:
    res = open('/tmp/myLinear_CUDA.txt', 'r').readlines()
    res.pop(0)
    count = 0
    s = 0
    for i in res:
        s += int(i)
        count += 1
    
    print('Avg CUDA forward time %.2f ms' % (s / count))

else:
    res = open('/tmp/myLinear_CPU.txt', 'r').readlines()
    res.pop(0)
    count = 0
    s = 0
    for i in res:
        s += int(i)
        count += 1

    print('Avg CPU forward time %.2f ms' % (s / count))

#print(weight)
#print(x)
#print(y)
