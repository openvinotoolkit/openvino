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
setup(name='llmdnn',
      ext_modules=[
        cpp_extension.CppExtension(
          'llmdnn',
          ['module.cpp', 'mha_gpt.cpp', '../../src/test_common.cpp'],
          extra_compile_args=[ '-fopenmp',
                              '-march=native',
                              #'-g'
                              ],
          #extra_link_args=['-lgomp'],
          include_dirs=['../../src',
                        '../../../include',
                        '../../../src'],
          library_dirs=[f'{sys.prefix}/lib',
                        '../../../../../../../../bin/intel64/Debug'],
          #runtime_library_dirs=[ f'{sys.prefix}/lib', ],
          libraries=['llmdnn',
                     'stdc++']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)}
      )