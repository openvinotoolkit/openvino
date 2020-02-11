/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "test_convolution_depthwise_forward_common.hpp"

namespace mkldnn {

using convolution_test = convolution_depthwise_test<uint8_t, int8_t, int32_t, float>;

TEST_P(convolution_test, TestConvolution)
{
}

#define EXPAND_FORMATS(src, weights, bias, dst) \
    { mkldnn::memory::format::src, mkldnn::memory::format::weights, \
    mkldnn::memory::format::bias, mkldnn::memory::format::dst }

#define FMT_WEIGHTS_BLOCKED8 OhIw8o4i
#define FMT_WEIGHTS_BLOCKED8_DW Goihw8g
#define FMT_WEIGHTS_BLOCKED16 OIhw4i16o4i
#define FMT_WEIGHTS_BLOCKED16_DW Goihw16g

#define ENGINE mkldnn::engine::kind::cpu
#define ALGORITHM mkldnn::convolution_direct

#define CONCAT_WITH_UNDERSCORE_(a,b) a ## _ ## b
#define CONCAT_WITH_UNDERSCORE(a,b) CONCAT_WITH_UNDERSCORE_(a,b)

#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test, ::testing::Values(__VA_ARGS__))

#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(CONCAT_WITH_UNDERSCORE(Convolution, \
        str), depthwise),  __VA_ARGS__)

#define EXPAND_ARGS(args) args

#define PARAMS(...) \
    EXPAND_ARGS(PARAMS_CONV(depthwise_scale_shift, __VA_ARGS__)), \
    EXPAND_ARGS(PARAMS_CONV(depthwise_prelu, __VA_ARGS__))

#define PARAMS_CONV(alg, src, weights, bias, dst, ...) \
    test_convolution_depthwise_params_t {alg,  ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), /* empty attributes */ {}, \
    {__VA_ARGS__} }

    INST_TEST_CASE(SimpleSmall,
        PARAMS(nhwc, oihw, x, nhwc,
               2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, oihw, x, nhwc,
               2, 1, 16, 13, 13, 48, 13, 13, 1, 1, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc,
               2, 64, 64, 16, 16, 64, 16, 16, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc,
               2, 32, 32, 9, 9, 32, 9, 9, 1, 1, 0, 0, 1, 1)
    );

    INST_TEST_CASE(SimpleSmall_Gemm,
        PARAMS(nhwc, hwio, x, nhwc,
               2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc,
               2, 1, 16, 13, 13, 48, 13, 13, 1, 1, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc,
               2, 64, 64, 16, 16, 64, 16, 16, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc,
               2, 32, 32, 9, 9, 32, 9, 9, 1, 1, 0, 0, 1, 1)
    );

    INST_TEST_CASE(SimpleSmall_Blocked8,
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8, x, nhwc,
               2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8, x, nhwc,
               2, 1, 16, 13, 13, 48, 13, 13, 1, 1, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8_DW, x, nhwc,
               2, 64, 64, 16, 16, 64, 16, 16, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8_DW, x, nhwc,
               2, 32, 32, 9, 9, 32, 9, 9, 1, 1, 0, 0, 1, 1)
    );

    INST_TEST_CASE(SimpleSmall_Blocked_Tail8,
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8, x, nhwc,
               2, 1, 15, 13, 13, 19, 11, 11, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8, x, nhwc,
               2, 1, 77, 13, 13, 91, 13, 13, 1, 1, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8_DW, x, nhwc,
               2, 21, 21, 16, 16, 21, 16, 16, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED8_DW, x, nhwc,
               2, 77, 77, 9, 9, 77, 9, 9, 1, 1, 0, 0, 1, 1)
    );

    INST_TEST_CASE(SimpleSmall_Blocked16,
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED16, x, nhwc,
               2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED16, x, nhwc,
               2, 1, 16, 13, 13, 48, 13, 13, 1, 1, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED16_DW, x, nhwc,
               2, 64, 64, 16, 16, 64, 16, 16, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, FMT_WEIGHTS_BLOCKED16_DW, x, nhwc,
               2, 32, 32, 9, 9, 32, 9, 9, 1, 1, 0, 0, 1, 1)
    );
}
