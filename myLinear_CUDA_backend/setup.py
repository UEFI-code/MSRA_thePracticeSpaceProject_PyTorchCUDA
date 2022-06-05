from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mySuperExtension',
    ext_modules=[
        CUDAExtension('myLinear_cuda', [
            'myLinear.cpp',
            'myLinearGPU.cu',
            'myLinearCPU.cpp'
        ]),
        CUDAExtension('myLinear_benchmark', [
            'myLinear_withBenchmark.cpp',
            'myLinearGPU.cu',
            'myLinearCPU.cpp'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
