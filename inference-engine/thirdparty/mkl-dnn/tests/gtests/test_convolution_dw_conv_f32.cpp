/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include "test_convolution_dw_conv_common.hpp"
namespace mkldnn {

using convolution_test = convolution_dw_conv_test<float, float, float, float>;

TEST_P(convolution_test, TestConvolutionDwConv)
{
}

#define FMT_BIAS x
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c

#define EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst) \
    { mkldnn::memory::format::src, mkldnn::memory::format::conv1_weights, mkldnn::memory::format::conv1_bias, \
    mkldnn::memory::format::conv2_weights, mkldnn::memory::format::conv2_bias, mkldnn::memory::format::dst }

#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o

#define FMT_WEIGHTS_DW_BLOCKED Goihw8g
#define FMT_WEIGHTS_DW_BLOCKED16 Goihw16g

#define ENGINE mkldnn::engine::kind::cpu
#define ALGORITHM mkldnn::convolution_direct

#define CONCAT_WITH_UNDERSCORE_(a,b) a ## _ ## b
#define CONCAT_WITH_UNDERSCORE(a,b) CONCAT_WITH_UNDERSCORE_(a,b)

#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test, ::testing::Values(__VA_ARGS__))

#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, \
        str), dw_conv),  __VA_ARGS__)

#define EXPAND_ARGS(args) args

#define PARAMS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst, ...) \
    test_convolution_dw_conv_params_t {ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst), {__VA_ARGS__} }

INST_TEST_CASE(Mobilenet_Blocked,
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 8, 19, 33,  56, 3, 3, 1, 1, 2, 2,  56, 3, 3, 1, 1, 1, 1), // 1_1
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 32, 19, 33,  56, 1, 1, 0, 0, 1, 1,  56, 3, 3, 1, 1, 2, 2), // 2_1
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 56, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 1, 1), // 2_2
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 112, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 2, 2), // 3_1
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 112, 4, 8,  208, 1, 1, 0, 0, 1, 1,  208, 3, 3, 1, 1, 1, 1),  // 3_2
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 208, 4, 8,  216, 1, 1, 0, 0, 1, 1,  216, 3, 3, 1, 1, 2, 2),  // 4_1
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 216, 2, 4,  328, 1, 1, 0, 0, 1, 1,  328, 3, 3, 1, 1, 1, 1),  // 4_2
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 328, 2, 4,  288, 1, 1, 0, 0, 1, 1,  288, 3, 3, 1, 1, 1, 1),  // 5_1
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 288, 2, 4,  288, 1, 1, 0, 0, 1, 1,  288, 3, 3, 1, 1, 1, 1),  // 5_2
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 288, 2, 4,  240, 1, 1, 0, 0, 1, 1,  240, 3, 3, 1, 1, 1, 1),  // 5_3
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 240, 2, 4,  264, 1, 1, 0, 0, 1, 1,  264, 3, 3, 1, 1, 1, 1)   // 5_4
);

}
