from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='myConv1D',
    ext_modules=[
        CUDAExtension('myConv1D', [
            'myConv1D.cpp',
            'myGPU.cu',
            'myCPU.cpp'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
