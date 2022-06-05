from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='myKakuritsu',
    ext_modules=[
        CUDAExtension('myKakuritsu_Linear', [
            'myKakuritsu.cpp',
            'myKakuritsuGPU.cu',
            'myKakuritsuCPU.cpp'
        ]),
        CUDAExtension('myKakuritsu_Benchmark', [
            'myKakuritsu_withBenchmark.cpp',
            'myKakuritsuGPU.cu',
            'myKakuritsuCPU.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
