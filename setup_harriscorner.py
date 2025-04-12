
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='harris_corner_detection',
    ext_modules=[
        CUDAExtension('hcd_on_gpu', [
            'hc/CornerDetector_cuda.cu',
            'hc/CornerDetector.cpp'
        ])
    ],
    cmdclass={'build_ext': BuildExtension})

# python setup_harriscorner.py build install
