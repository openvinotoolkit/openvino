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

using convolution_test_3d = convolution_forward_test_3d<uint8_t, int8_t, int32_t, int32_t>;
using convolution_test_3d_s8 = convolution_forward_test_3d<int8_t, int8_t, int32_t, int32_t>;

TEST_P(convolution_test_3d, TestConvolution)
{
}

TEST_P(convolution_test_3d_s8, TestConvolution)
{
}

#define _3D
#define U8S8
#define S8S8
#define DIRECTION_FORWARD
#include "convolution_common.h"

INST_TEST_CASE_3D(Simple_Gemm_u8s8s32_3d,
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
                            2, 1, 15, 3, 3, 3, 37, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 14, 4, 4, 4, 1, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 33, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 19, 2, 2, 2, 22, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1)
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

INST_TEST_CASE_3D_SIGNED(SimpleSmall_NCDHW,
                  PARAMS_3D(ndhwc, dhwio_s8s8, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(ndhwc, dhwio_s8s8, FMT_BIAS, ndhwc,
                            2, 1, 4, 4, 4, 4, 6, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(ndhwc, dhwigo_s8s8, FMT_BIAS, ndhwc,
                            2, 2, 4, 4, 4, 4, 6, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(ndhwc, dhwigo_s8s8, FMT_BIAS, ndhwc,
                            2, 2, 4, 4, 4, 4, 6, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1)
);

INST_TEST_CASE_3D_SIGNED(SimpleSmall_Blocked_Padded_Channels,
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 5, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 15, 3, 3, 3, 37, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 14, 4, 4, 4, 1, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 33, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 19, 2, 2, 2, 22, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1)
);

INST_TEST_CASE_3D_SIGNED(SimpleSmall_Blocked_1x1_Padded_Channels,
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 3, 13, 13, 13, 35, 13, 13, 13, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 7, 3, 3, 3, 11, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 1, 4, 4, 4, 58, 4, 4, 4, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 27, 3, 3, 3, 33, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                  PARAMS_3D(FMT_DATA_BLOCKED_3D, FMT_WEIGHTS_BLOCKED_SIGNED_3D, FMT_BIAS, FMT_DATA_BLOCKED_3D,
                            2, 1, 81, 2, 2, 2, 81, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1)
);

}
