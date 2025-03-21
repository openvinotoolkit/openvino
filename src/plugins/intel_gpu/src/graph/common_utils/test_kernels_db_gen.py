#!/usr/bin/python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import os, sys
import argparse
from kernels_db_gen import Code2CHeaders

class TestOpenCLCodePreprocessing(unittest.TestCase):
    def setUp(self):
        self.kernels_folder = 'test_kernels'
        self.headers_folder = os.path.join(self.kernels_folder, "include")
        self.batch_headers_folder = os.path.join(self.kernels_folder, "include/batch_headers")

        self.test_kernel_path = os.path.join(self.kernels_folder, 'test_kernel.cl')
        self.batch_header1_path = os.path.join(self.batch_headers_folder, 'batch_header1.cl')
        self.batch_header2_path = os.path.join(self.batch_headers_folder, 'batch_header2.cl')
        self.header_path = os.path.join(self.headers_folder, 'header.cl')
        self.no_opt_header_path = os.path.join(self.headers_folder, 'no_opt_header.cl')

        self.input_code = '''
#include "include/batch_headers/batch_header1.cl"
#include "include/header.cl"
#include "include/header.cl"

#define SOME_MACRO_PARAM 10
#include "include/no_opt_header.cl" [[no_opt]]
SOME_MACRO_WITH_PARAM(1)
#undef SOME_MACRO_PARAM

#define SOME_MACRO_PARAM 20
#include "include/no_opt_header.cl" [[no_opt]]
SOME_MACRO_WITH_PARAM(1)
#undef SOME_MACRO_PARAM

#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)

#define USED_SIMPLE_MACRO 10
#define UNUSED_SIMPLE_MACRO 20
    #     define      USED_MACRO_WITH_SPACES      10
#define USED_MULTI_LINE_MACRO \
100 \
101
#define UNUSED_MULTI_LINE_MACRO \
200 \
201
#define USED_MACRO_CONCAT_NAME 1000
#define USED_MACRO_CONCAT_NAME_NESTED 1000
#define UNUSED_MACRO_WITH_UNDEF smth
#define USED_MACRO_FUNC(a, b) a > b
#define UNUSED_MACRO_FUNC(a, b) a < b

// some single line comment
__kernel void some_kernel(int arg1,
#ifdef USED_SIMPLE_MACRO
int arg2
#else
int arg3
#endif
) {
    int a = USED_SIMPLE_MACRO     +     USED_MACRO_WITH_SPACES + USED_MULTI_LINE_MACRO;
    int b = USED_MACRO_FUNC(a, 10) ? 1 : 0;




/*
multi-line comment
*/
    int c = CAT(USED_MACRO_CONCAT, _NAME);  // some inline comment
    int d = CAT(CAT(USED_MACRO_CONCAT, _NAME), _NESTED);
    int e = USED_REGULAR_HEADER_MACRO;
}

#undef UNUSED_MACRO_WITH_UNDEF
'''
        self.batch_header1 = '''
#define BATCH_HEADER1_MACRO 1
'''
        self.batch_header2 = '''
#include "include/batch_headers/batch_header1.cl"
#define BATCH_HEADER2_MACRO 2
'''
        self.regular_header = '''
#include "include/batch_headers/batch_header1.cl"
#include "include/batch_headers/batch_header2.cl"
#define UNUSED_REGULAR_HEADER_MACRO 1
#define USED_REGULAR_HEADER_MACRO 2
'''
        self.no_opt_header = '''
#define SOME_MACRO_WITH_PARAM(a) a + SOME_MACRO_PARAM

#undef SOME_MACRO_WITH_PARAM
'''

        self.expected_batch_header1 = '''std::make_pair<std::string_view, std::string_view>("batch_header1", R"__krnl(#define BATCH_HEADER1_MACRO 1)__krnl"),
'''

        self.expected_batch_header2 = '''\
std::make_pair<std::string_view, std::string_view>("batch_header2", R"__krnl(#include "include/batch_headers/batch_header1.cl"
#define BATCH_HEADER2_MACRO 2)__krnl"),
'''

        self.expected_source = '''std::make_pair<std::string_view, std::string_view>("test_kernel", R"__krnl(#include "include/batch_headers/batch_header1.cl"
#include "include/batch_headers/batch_header2.cl"
#define USED_REGULAR_HEADER_MACRO 2
#define SOME_MACRO_PARAM 10
#define SOME_MACRO_WITH_PARAM(a) a+SOME_MACRO_PARAM
#undef SOME_MACRO_WITH_PARAM
SOME_MACRO_WITH_PARAM(1)
#undef SOME_MACRO_PARAM
#define SOME_MACRO_PARAM 20
#define SOME_MACRO_WITH_PARAM(a) a+SOME_MACRO_PARAM
#undef SOME_MACRO_WITH_PARAM
SOME_MACRO_WITH_PARAM(1)
#undef SOME_MACRO_PARAM
#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)
#define USED_SIMPLE_MACRO 10
#define USED_MACRO_WITH_SPACES 10
#define USED_MULTI_LINE_MACRO 100 101
#define USED_MACRO_CONCAT_NAME 1000
#define USED_MACRO_CONCAT_NAME_NESTED 1000
#define USED_MACRO_FUNC(a,b) a>b
__kernel void some_kernel(int arg1,
#ifdef USED_SIMPLE_MACRO
int arg2
#else
int arg3
#endif
){
int a=USED_SIMPLE_MACRO+USED_MACRO_WITH_SPACES+USED_MULTI_LINE_MACRO;
int b=USED_MACRO_FUNC(a,10) ? 1 : 0;
int c=CAT(USED_MACRO_CONCAT,_NAME);
int d=CAT(CAT(USED_MACRO_CONCAT,_NAME),_NESTED);
int e=USED_REGULAR_HEADER_MACRO;
}
#undef CAT
#undef USED_MACRO_CONCAT_NAME
#undef USED_MACRO_CONCAT_NAME_NESTED
#undef USED_MACRO_FUNC
#undef USED_MACRO_WITH_SPACES
#undef USED_MULTI_LINE_MACRO
#undef USED_REGULAR_HEADER_MACRO
#undef USED_SIMPLE_MACRO
#undef _CAT)__krnl"),
'''

    def test_opencl_code_preprocessing(self):
        def write_to_file(file_path, content):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(content)

        write_to_file(self.test_kernel_path, self.input_code)
        write_to_file(self.batch_header1_path, self.batch_header1)
        write_to_file(self.batch_header2_path, self.batch_header2)
        write_to_file(self.header_path, self.regular_header)
        write_to_file(self.no_opt_header_path, self.no_opt_header)

        processor = Code2CHeaders(self.kernels_folder, self.headers_folder, "ocl")
        ocl_sources, ocl_headers = processor.generate()

        self.assertEqual(ocl_sources[0], self.expected_source)
        self.assertEqual(ocl_headers[0], self.expected_batch_header1)
        self.assertEqual(ocl_headers[1], self.expected_batch_header2)

        os.remove(self.test_kernel_path)
        os.remove(self.batch_header1_path)
        os.remove(self.batch_header2_path)
        os.remove(self.header_path)
        os.remove(self.no_opt_header_path)

if __name__ == '__main__':
    unittest.main()
