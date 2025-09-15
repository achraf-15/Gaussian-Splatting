from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gaussian_renderer_cuda',
    ext_modules=[
        CUDAExtension(
            name='gaussian_renderer_cuda',
            sources=['gaussian_renderer.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
