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

TEST_P(binary_convolution_test, TestBinaryConvolutionDwConvBinarization)
{
}

#define BIN
#define WITH_DW_CONV
#define WITH_BINARIZATION
#define DIRECTION_FORWARD
#include "convolution_common.h"

#define PARAMS_WITH_BINARIZATION(...) \
    EXPAND_ARGS(PARAMS(binarization_depthwise, __VA_ARGS__))

INST_TEST_CASE(SimpleSmall_Blocked,
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           1, 19, 5, 5,  77, 1, 1, 0, 0, 1, 1,  77, 3, 3, 1, 1, 1, 1)
);

INST_TEST_CASE(Mobilenet_Blocked,
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 8, 19, 33,  56, 3, 3, 1, 1, 2, 2,  56, 3, 3, 1, 1, 1, 1), // 1_1
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 56, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 1, 1), // 2_2
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 112, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 2, 2), // 3_1
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 288, 2, 4,  240, 1, 1, 0, 0, 1, 1,  240, 3, 3, 1, 1, 1, 1)  // 5_3
);

INST_TEST_CASE(SimpleSmall_Blocked16,
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           1, 19, 5, 5,  77, 1, 1, 0, 0, 1, 1,  77, 3, 3, 1, 1, 1, 1)
);

INST_TEST_CASE(Mobilenet_Blocked16,
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 8, 19, 33,  56, 3, 3, 1, 1, 2, 2,  56, 3, 3, 1, 1, 1, 1), // 1_1
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 56, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 1, 1), // 2_2
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 112, 9, 16,  112, 1, 1, 0, 0, 1, 1,  112, 3, 3, 1, 1, 2, 2), // 3_1
    PARAMS_WITH_BINARIZATION(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_WEIGHTS_DW_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED,
           2, 288, 2, 4,  240, 1, 1, 0, 0, 1, 1,  240, 3, 3, 1, 1, 1, 1)  // 5_3
);

}
