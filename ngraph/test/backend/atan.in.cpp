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
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, atan)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f});
    test_case.add_expected_output<float>(shape,
                                         {-1.32581766f,
                                          -1.10714872f,
                                          -0.78539816f,
                                          -0.46364761f,
                                          -0.24497866f,
                                          0.00000000f,
                                          0.24497866f,
                                          0.46364761f,
                                          0.78539816f,
                                          1.10714872f,
                                          1.32581766f});
    test_case.run();
}
