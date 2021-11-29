// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_1d)
{
    const Shape a_shape{3};
    const auto A = make_shared<op::Parameter>(element::f32, a_shape);

    const auto alpha = op::Constant::create(element::f32, Shape{}, {0.5f});
    const auto beta = op::Constant::create(element::f32, Shape{}, {0.6f});

    const auto R = make_shared<op::v0::HardSigmoid>(A, alpha, beta);
    const auto f = make_shared<Function>(R, ParameterVector{A});

    std::vector<float> a{-1.0f, 0.0f, 1.0f};

    EXPECT_EQ(R->get_output_shape(0), a_shape);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>({a});
    test_case.add_expected_output<float>({0.1f, 0.6f, 1.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_2d)
{
    const Shape a_shape{2, 5};
    const auto A = make_shared<op::Parameter>(element::f32, a_shape);

    const auto alpha = op::Constant::create(element::f32, Shape{}, {0.2f});
    const auto beta = op::Constant::create(element::f32, Shape{}, {0.5f});

    const auto R = make_shared<op::v0::HardSigmoid>(A, alpha, beta);
    const auto f = make_shared<Function>(R, ParameterVector{A});

    std::vector<float> a{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 0.5f, -0.2f, 6.0f, 8.0f, 0.1f};

    EXPECT_EQ(R->get_output_shape(0), a_shape);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(
        a_shape, {0.0f, 0.3f, 0.5f, 0.7f, 1.0f, 0.6f, 0.46f, 1.0f, 1.0f, 0.52f});
    test_case.run();
}
