# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from setuptools import setup, Extension
from torch.utils import cpp_extension
import sys
import os

'''
using intel compiler:
source ~/intel/oneapi/setvars.sh
export CXX=icx
export CC=icx
'''
debug = False
if 'DEBUG_EXT' in os.environ:
  debug = True if os.environ['DEBUG_EXT'] == '1' else False
extra_args = ['-fopenmp',
              '-march=native']
llmdnn_lib_dir = f'{os.getcwd()}/../../../../../../../../bin/intel64/Release'
if debug:
  llmdnn_lib_dir = f'{os.getcwd()}/../../../../../../../../bin/intel64/Debug'
  extra_args += ['-g', '-O0']
  print('install debug version')
else:
  print('install release version')

setup(name='llmdnn',
      ext_modules=[
        cpp_extension.CppExtension(
          'llmdnn',
          ['module.cpp', f'../../src/test_common.cpp',
           'mha_gpt.cpp',
           'emb_gpt.cpp',
           'attn_gpt.cpp',
           ],
          extra_compile_args=extra_args,
          include_dirs=[f'{os.getcwd()}/../../src',
                        f'{os.getcwd()}/../../../include',
                        f'{os.getcwd()}/../../../src'],
          library_dirs=[f'{sys.prefix}/lib',
                        llmdnn_lib_dir],
          libraries=['llmdnn',
                     'stdc++']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )