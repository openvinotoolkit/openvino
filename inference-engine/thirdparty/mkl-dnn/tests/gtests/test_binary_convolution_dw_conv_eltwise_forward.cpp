/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include "test_binary_convolution_dw_conv_forward_common.hpp"

namespace mkldnn {

using binary_convolution_test = binary_convolution_forward_test;

TEST_P(binary_convolution_test, TestBinaryConvolutionDwConvEltwise)
{
}

#define BIN
#define WITH_DW_CONV
#define WITH_ELTWISE
#define DIRECTION_FORWARD
#include "convolution_common.h"

#define PARAMS_WITH_ELTIWSE(...) \
    EXPAND_ARGS(PARAMS(eltwise_relu, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_elu, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_tanh, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_square, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_abs, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_sqrt, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_linear, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_bounded_relu, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_soft_relu, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_logistic, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS(eltwise_clamp, __VA_ARGS__))

INST_TEST_CASE(Mobilenet_Blocked,
    PARAMS_WITH_ELTIWSE(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           1, 7, 10, 10,  37, 1, 1, 0, 0, 1, 1,  37, 3, 3, 1, 1, 2, 2)
);

INST_TEST_CASE(Mobilenet_Blocked16,
    PARAMS_WITH_ELTIWSE(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           1, 7, 10, 10,  37, 1, 1, 0, 0, 1, 1,  37, 3, 3, 1, 1, 2, 2)
);

}
