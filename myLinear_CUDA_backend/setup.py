from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mylinear_cuda',
    ext_modules=[
        CUDAExtension('myLinear_cuda', [
            'myLinear.cpp',
            'myLinearGPU.cu',
            'myLinearCPU.cpp'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
