# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from setuptools import setup, Extension
from torch.utils import cpp_extension
import sys

'''
using intel compiler:
source ~/intel/oneapi/setvars.sh
export CXX=icx
export CC=icx
'''
debug = True
extra_args = ['-fopenmp',
              '-march=native']
llmdnn_lib_dir = '../../../../../../../../bin/intel64/Release'
if debug:
  llmdnn_lib_dir = '../../../../../../../../bin/intel64/Debug'
  extra_args += ['-g', '-O0']
setup(name='llmdnn',
      ext_modules=[
        cpp_extension.CppExtension(
          'llmdnn',
          ['module.cpp', 'mha_gpt.cpp', '../../src/test_common.cpp'],
          extra_compile_args=extra_args,
          #extra_link_args=['-lgomp'],
          include_dirs=['../../src',
                        '../../../include',
                        '../../../src'],
          library_dirs=[f'{sys.prefix}/lib',
                        llmdnn_lib_dir],
          #runtime_library_dirs=[ f'{sys.prefix}/lib', ],
          libraries=['llmdnn',
                     'stdc++']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)}
      )