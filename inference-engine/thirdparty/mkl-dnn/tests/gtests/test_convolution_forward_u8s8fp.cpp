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
#include "test_convolution_forward_common.hpp"
namespace mkldnn {

using convolution_test = convolution_forward_test<uint8_t, int8_t,
                                                  int32_t, float>;

TEST_P(convolution_test, TestConvolution)
{
}

//#define TEST_PARAM_ATTR
#define U8S8
#define DIRECTION_FORWARD
#include "convolution_common.h"

INST_TEST_CASE(SimpleSmall_Blocked_Padded_Channels,
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 7, 3, 3, 5, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 15, 3, 3, 37, 4, 4, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 14, 4, 4, 1, 4, 4, 3, 3, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 7, 3, 3, 33, 3, 3, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 19, 2, 2, 22, 2, 2, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 126, 13, 13, 126, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 77, 13, 13, 99, 11, 11, 3, 3, 0, 0, 1, 1)
);

INST_TEST_CASE(SimpleSmall_Blocked_1x1_Padded_Channels,
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 13, 13, 35, 13, 13, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 7, 3, 3, 11, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 1, 4, 4, 58, 4, 4, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 27, 3, 3, 33, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 81, 2, 2, 81, 2, 2, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 126, 13, 13, 13, 13, 13, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 111, 13, 13, 71, 13, 13, 1, 1, 0, 0, 1, 1)
);

INST_TEST_CASE(SimpleSmall_Depthwise_Blocked_Padded_Channels,
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 126, 126, 10, 10, 126, 10, 10, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 77, 77, 9, 9, 77, 2, 2, 5, 5, 0, 0, 3, 3),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 68, 68, 26, 26, 68, 13, 13, 4, 4, 1, 1, 2, 2),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 33, 33, 111, 111, 33, 112, 112, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 111, 111, 1, 2, 111, 1, 1, 3, 3, 1, 1, 1, 2),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 29, 29, 16, 32, 29, 16, 18, 3, 3, 1, 2, 1, 2),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 53, 53, 32, 16, 53, 16, 14, 3, 3, 1, 0, 2, 1),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 13, 13, 32, 16, 13, 18, 16, 3, 3, 2, 1, 2, 1),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 9, 9, 500, 500, 9, 698, 698, 3, 3, 100, 100, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, Goihw8g, FMT_BIAS, FMT_DATA_BLOCKED,
        1, 2, 2, 500, 500, 2, 698, 698, 3, 3, 100, 100, 1, 1)
);

//#undef TEST_PARAM_ATTR

}
