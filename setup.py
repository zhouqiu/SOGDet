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


# def parse_requirements(fname='requirements.txt', with_version=True):
#     """Parse the package dependencies listed in a requirements file but strips
#     specific versioning information.

#     Args:
#         fname (str): path to requirements file
#         with_version (bool, default=False): if True include version specs

#     Returns:
#         list[str]: list of requirements items

#     CommandLine:
#         python -c "import setup; print(setup.parse_requirements())"
#     """
#     import re
#     import sys
#     from os.path import exists
#     require_fpath = fname

#     def parse_line(line):
#         """Parse information from a line in a requirements text file."""
#         if line.startswith('-r '):
#             # Allow specifying requirements in other files
#             target = line.split(' ')[1]
#             for info in parse_require_file(target):
#                 yield info
#         else:
#             info = {'line': line}
#             if line.startswith('-e '):
#                 info['package'] = line.split('#egg=')[1]
#             else:
#                 # Remove versioning from the package
#                 pat = '(' + '|'.join(['>=', '==', '>']) + ')'
#                 parts = re.split(pat, line, maxsplit=1)
#                 parts = [p.strip() for p in parts]

#                 info['package'] = parts[0]
#                 if len(parts) > 1:
#                     op, rest = parts[1:]
#                     if ';' in rest:
#                         # Handle platform specific dependencies
#                         # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
#                         version, platform_deps = map(str.strip,
#                                                      rest.split(';'))
#                         info['platform_deps'] = platform_deps
#                     else:
#                         version = rest  # NOQA
#                     info['version'] = (op, version)
#             yield info

#     def parse_require_file(fpath):
#         with open(fpath, 'r') as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 if line and not line.startswith('#'):
#                     for info in parse_line(line):
#                         yield info

#     def gen_packages_items():
#         if exists(require_fpath):
#             for info in parse_require_file(require_fpath):
#                 parts = [info['package']]
#                 if with_version and 'version' in info:
#                     parts.extend(info['version'])
#                 if not sys.version.startswith('3.4'):
#                     # apparently package_deps are broken in 3.4
#                     platform_deps = info.get('platform_deps')
#                     if platform_deps is not None:
#                         parts.append(';' + platform_deps)
#                 item = ''.join(parts)
#                 yield item

#     packages = list(gen_packages_items())
#     return packages


# def add_mim_extention():
#     """Add extra files that are required to support MIM into the package.

#     These files will be added by creating a symlink to the originals if the
#     package is installed in `editable` mode (e.g. pip install -e .), or by
#     copying from the originals otherwise.
#     """

#     # parse installment mode
#     if 'develop' in sys.argv:
#         # installed by `pip install -e .`
#         mode = 'symlink'
#     elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
#         # installed by `pip install .`
#         # or create source distribution by `python setup.py sdist`
#         mode = 'copy'
#     else:
#         return

#     filenames = ['tools', 'configs', 'model-index.yml']
#     repo_path = osp.dirname(__file__)
#     mim_path = osp.join(repo_path, 'mmdet3d', '.mim')
#     os.makedirs(mim_path, exist_ok=True)

#     for filename in filenames:
#         if osp.exists(filename):
#             src_path = osp.join(repo_path, filename)
#             tar_path = osp.join(mim_path, filename)

#             if osp.isfile(tar_path) or osp.islink(tar_path):
#                 os.remove(tar_path)
#             elif osp.isdir(tar_path):
#                 shutil.rmtree(tar_path)

#             if mode == 'symlink':
#                 src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
#                 os.symlink(src_relpath, tar_path)
#             elif mode == 'copy':
#                 if osp.isfile(src_path):
#                     shutil.copyfile(src_path, tar_path)
#                 elif osp.isdir(src_path):
#                     shutil.copytree(src_path, tar_path)
#                 else:
#                     warnings.warn(f'Cannot copy file {src_path}.')
#             else:
#                 raise ValueError(f'Invalid mode {mode}')


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
            )
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
