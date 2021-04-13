// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include "util/random.hpp"

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

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d) 
{
    Shape shape_a{2, 6};
    Shape shape_b{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         1, 2, -3, -4, 5, 6};
    std::vector<float> b{2, 2, 2, 2, 2, 2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -6, -8, 5, 6,
                                                   1, 2, -6, -8, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d) 
{
    Shape shape_a{6};
    Shape shape_b{1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> b{2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -6, -8, 5, 6});
    test_case.run();
}