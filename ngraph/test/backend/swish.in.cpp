// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, swish_2D_with_beta0_6)
{
    Shape in_shape{2, 4};
    element::Type et = element::f32;
    auto beta = 0.6f;

    auto args0 = make_shared<op::Parameter>(et, in_shape);
    auto args1 = make_shared<op::Parameter>(et, Shape{});
    auto swish = make_shared<op::v4::Swish>(args0, args1);
    auto f = make_shared<Function>(swish, ParameterVector{args0, args1});

    vector<vector<float>> in_vec{vector<float>{0.4, -5.7, -6, 3, -0.9, 23, 5, 3.3} , vector<float>{beta}};
    vector<float> out_vec{in_vec[0]};
    std::transform(out_vec.begin(), out_vec.end(), out_vec.begin(), [&beta](float x) -> float { return (x / (1.0f + std::exp(x * beta * -1.0f)));});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>(in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, swish_2D_without_beta)
{
    Shape in_shape{2, 3};
    element::Type et = element::f32;

    auto args0 = make_shared<op::Parameter>(et, in_shape);
    auto swish = make_shared<op::v4::Swish>(args0);
    auto f = make_shared<Function>(swish, ParameterVector{args0});

    vector<float> in_vec{1, 8, -8, 17, -0.5, -1};
    vector<float> out_vec{in_vec};
    std::transform(out_vec.begin(), out_vec.end(), out_vec.begin(), [](float x) -> float { return (x / (1.0f + std::exp(x * -1.0f)));});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, swish_4D_with_beta0_33)
{
    Shape in_shape{2, 2, 1, 2};
    element::Type et = element::f32;
    auto beta = 0.33f;

    auto args0 = make_shared<op::Parameter>(et, in_shape);
    auto args1 = make_shared<op::Parameter>(et, Shape{});
    auto swish = make_shared<op::v4::Swish>(args0, args1);
    auto f = make_shared<Function>(swish, ParameterVector{args0, args1});

    vector<vector<float>> in_vec{vector<float>{0.1, 0.6, 20, -7, -5.3, 3.5, -9, 11} , vector<float>{beta}};
    vector<float> out_vec{in_vec[0]};
    std::transform(out_vec.begin(), out_vec.end(), out_vec.begin(), [&beta](float x) -> float { return (x / (1.0f + std::exp(x * beta * -1.0f)));});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>(in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run();
}
