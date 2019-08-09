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
#include "test_deformable_convolution_forward_common.hpp"
namespace mkldnn {

using deformable_convolution_test = deformable_convolution_forward_test<float, float, float, float, float>;

TEST_P(deformable_convolution_test, TestDeformableConvolution)
{
}

#define DEF
#define FP32
#define DIRECTION_FORWARD
#include "convolution_common.h"

INST_TEST_CASE(SimpleSmall_Blocked8,
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),

    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),

    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw8i8o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2)
);

INST_TEST_CASE(SimpleSmall_Blocked16,
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 1, 10, 10, 1, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 1, 17, 10, 10, 33, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),

    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 4, 64, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2),

    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 10, 10, 1, 1, 0, 0, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 1, 1),
    PARAMS(nhwc, nchw, OIhw16i16o, x, nhwc,
        2, 1, 32, 32, 10, 10, 48, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2)
);

}
