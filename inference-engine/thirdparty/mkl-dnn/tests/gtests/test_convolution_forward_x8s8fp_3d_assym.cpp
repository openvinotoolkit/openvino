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
#include "test_convolution_forward_common_3d.hpp"

namespace mkldnn {

using convolution_test_3d = convolution_forward_test_3d<uint8_t, int8_t, int32_t, float>;

TEST_P(convolution_test_3d, TestConvolution)
{
}

#define _3D
#define U8S8
#define DIRECTION_FORWARD
#define ASYMMETRIC_QUANTIZATION
#include "convolution_common.h"

INST_TEST_CASE_3D(Simple_Gemm_u8s8fp_3d,
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 1, 1, 1, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              1, 1, 2, 2, 1, 5, 1, 2, 1, 4, 1, 1, 2, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              1, 1, 2, 4, 4, 5, 1, 2, 5, 3, 2, 2, 3, 0, 1, 2, 2, 1, 3),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              1, 1, 2, 4, 4, 5, 1, 1, 5, 5, 2, 2, 3, 0, 1, 2, 1, 1, 1, 2, 0, 1),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              1, 1, 4, 4, 4, 4, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwigo, FMT_BIAS, ndhwc,
              1, 2, 2, 2, 3, 2, 2, 1, 2, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwigo, FMT_BIAS, ndhwc,
              1, 2, 4, 2, 3, 2, 2, 1, 2, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwigo, FMT_BIAS, ndhwc,
              1, 2, 8, 2, 3, 2, 4, 1, 2, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
              2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(ndhwc, dhwigo, FMT_BIAS, ndhwc,
              1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1)
);

INST_TEST_CASE_3D(SimpleSmall_NCDHW,
                  PARAMS_3D(ncdhw, oidhw, FMT_BIAS, ncdhw,
                            2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(ncdhw, oidhw, FMT_BIAS, ncdhw,
                            2, 1, 4, 4, 4, 4, 6, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(ndhwc, oidhw, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(ndhwc, oidhw, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 2, 2, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(ndhwc, dhwio, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1)
);

INST_TEST_CASE_3D(SimpleSmall_Blocked_Padded_Channels,
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 5, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 16, 3, 3, 3, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 14, 4, 4, 4, 1, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 33, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 19, 2, 2, 2, 22, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_G_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 4, 32, 10, 10, 10, 64, 10, 10, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1)
);

INST_TEST_CASE_3D(SimpleSmall_Blocked_1x1_Padded_Channels,
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 3, 13, 13, 13, 35, 13, 13, 13, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 11, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 1, 4, 4, 4, 58, 4, 4, 4, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 27, 3, 3, 3, 33, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 81, 2, 2, 2, 81, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1)
);

INST_TEST_CASE_3D(SimpleSmall_Depthwise_Blocked_Padded_Channels,
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 8, 8, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 16, 16, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 9, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 126, 126, 10, 10, 10, 126, 10, 10, 10, 1, 3, 3, 0, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 3, 3, 10, 10, 10, 3, 10, 10, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 8, 8, 10, 10, 10, 8, 10, 10, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 9, 9, 1, 2, 2, 9, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 9, 9, 2, 1, 1, 9, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 24, 24, 2, 2, 2, 24, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 24, 24, 6, 6, 6, 24, 6, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 126, 126, 10, 10, 10, 126, 10, 10, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 77, 77, 9, 9, 9, 77, 2, 2, 2, 5, 5, 5, 0, 0, 0, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 68, 68, 26, 26, 26, 68, 13, 13, 13, 4, 4, 4, 1, 1, 1, 2, 2, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 33, 33, 21, 21, 21, 33, 22, 22, 22, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 111, 111, 2, 1, 2, 111, 1, 1, 1, 3, 3, 3, 1, 1, 1, 2, 1, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 29, 29, 8, 16, 32, 29, 8, 16, 18, 3, 3, 3, 1, 1, 2, 1, 1, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, Goidhw8g, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            1, 53, 53, 32, 32, 16, 53, 16, 16, 14, 3, 3, 3, 1, 1, 0, 2, 2, 1)
);

INST_TEST_CASE_3D(Simple_Blocked16,
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 13, 13, 13, 32, 12, 12, 12, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 3, 3, 3, 32, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 4, 4, 4, 32, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 3, 3, 3, 32, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 2, 2, 2, 32, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 13, 13, 13, 48, 13, 13, 13, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 13, 13, 13, 48, 11, 11, 11, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 8, 8, 8, 48, 5, 5, 5, 4, 4, 4, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 7, 7, 7, 48, 10, 10, 10, 4, 4, 4, 3, 3, 3, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 1, 1, 1, 48, 2, 2, 2, 4, 4, 4, 2, 2, 2, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 21, 21, 21, 48, 7, 7, 7, 5, 5, 5, 1, 1, 1, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 18, 18, 18, 48, 5, 5, 5, 6, 6, 6, 2, 2, 2, 4, 4, 4),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 6, 6, 6, 48, 2, 2, 2, 3, 3, 3, 0, 0, 0, 2, 2, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 9, 9, 9, 48, 2, 2, 2, 5, 5, 5, 0, 0, 0, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_G_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 4, 64, 10, 10, 10, 128, 10, 10, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1)
);

INST_TEST_CASE_3D(Simple_Dilated_Blocked16,
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 17, 17, 17, 32, 17, 17, 17, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 11, 11, 11, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 9, 9, 9, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 7, 7, 7, 3, 3, 3, 0, 0, 0, 1, 1, 1, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 7, 7, 7, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 6, 6, 6, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 5, 5, 5, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 9, 9, 9, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, FMT_WEIGHTS_BLOCKED16_3D, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 1, 32, 15, 15, 15, 32, 3, 3, 3, 5, 5, 5, 1, 1, 1, 2, 2, 2, 2, 2, 2)
);

INST_TEST_CASE_3D(Simple_Depthwise_Blocked16,
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, Goidhw16g, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, Goidhw16g, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 32, 32, 9, 9, 9, 32, 2, 2, 2, 5, 5, 5, 0, 0, 0, 3, 3, 3),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, Goidhw16g, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 64, 64, 26, 26, 26, 64, 13, 13, 13, 4, 4, 4, 1, 1, 1, 2, 2, 2),
                  PARAMS_3D(FMT_DATA_BLOCKED16_3D, Goidhw16g, FMT_BIAS, FMT_DATA_BLOCKED16_3D,
                            2, 32, 32, 11, 11, 11, 32, 12, 12, 12, 1, 1, 1, 0, 0, 0, 1, 1, 1)
);

}
