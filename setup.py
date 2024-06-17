from setuptools import find_packages, setup

import os
import shutil
import sys
import torch
import warnings
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


# def readme():
#     with open('README.md', encoding='utf-8') as f:
#         content = f.read()
#     return content


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)




if __name__ == '__main__':
    # add_mim_extention()
    setup(
        name='SOGDet',
        version='0.0.1',
        description=(''),
        author='SOGDet Authors',
        author_email='xxx@gmail.com',
        url='xxx',
        packages=find_packages(),
        include_package_data=True,
        package_data={'mmdet3d_plugin.ops': ['*/*.so']},
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d_plugin.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name='iou3d_cuda',
                module='mmdet3d_plugin.ops.iou3d',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
