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
#include "test_convolution_forward_common_3d.hpp"
#include "test_convolution_forward_common.hpp"
namespace mkldnn {

using convolution_test = convolution_forward_test<float, float, float, float>;
using convolution_test_3d = convolution_forward_test_3d<float, float, float, float>;

TEST_P(convolution_test_3d, TestConvolution)
{
}

#define FP32
#define DIRECTION_FORWARD
#include "convolution_common.h"

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

INST_TEST_CASE_3D(SimpleSmall_Blocked,
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 13, 13, 13, 32, 12, 12, 12, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 3, 3, 3, 32, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 4, 4, 4, 32, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 3, 3, 3, 32, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 2, 2, 2, 32, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 13, 13, 13, 48, 13, 13, 13, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw8c, OIdhw8i8o, FMT_BIAS, nCdhw8c,
        2, 1, 32, 13, 13, 13, 48, 11, 11, 11, 3, 3, 3, 0, 0, 0, 1, 1, 1)
);

INST_TEST_CASE_3D(SimpleSmall_Blocked16,
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 13, 13, 13, 32, 12, 12, 12, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 3, 3, 3, 32, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 4, 4, 4, 32, 4, 4, 4, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 3, 3, 3, 32, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 2, 2, 2, 32, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 13, 13, 13, 48, 13, 13, 13, 3, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS_3D(nCdhw16c, OIdhw16i16o, FMT_BIAS, nCdhw16c,
        2, 1, 32, 13, 13, 13, 48, 11, 11, 11, 3, 3, 3, 0, 0, 0, 1, 1, 1)
);

}
