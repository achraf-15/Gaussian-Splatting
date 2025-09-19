from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_engine",  # name of the package
    packages=["cuda_engine"],  # declare package
    ext_modules=[
        CUDAExtension(
            name="cuda_engine.gaussian_renderer_cuda",  
            sources=["cuda_engine/gaussian_renderer.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

