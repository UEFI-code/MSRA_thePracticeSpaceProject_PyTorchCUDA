from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mylinear_cuda',
    ext_modules=[
        CUDAExtension('myLinear_cuda', [
            'myLinear.cpp',
            'myLinear.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
