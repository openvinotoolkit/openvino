// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, sin)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{0.00000000f,
                                                0.24740396f,
                                                -0.24740396f,
                                                0.47942554f,
                                                -0.47942554f,
                                                0.84147098f,
                                                -0.84147098f,
                                                0.90929743f,
                                                -0.90929743f,
                                                -0.75680250f,
                                                0.75680250f},
                                  read_vector<float>(result)));
}
