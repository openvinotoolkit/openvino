// Copyright (C) 2021 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, selu_2Dfprop)
{
    Shape rt_shape{2};
    Shape c_shape{1};
    element::Type et = element::f32;

    auto input = make_shared<op::Parameter>(et, rt_shape);
    auto alpha = op::Constant::create(et, c_shape, {1.67326324});
    auto lambda = op::Constant::create(et, c_shape, {1.05070098});
    auto selu = make_shared<op::v0::Selu>(input, alpha, lambda);
    auto f = make_shared<Function>(selu, ParameterVector{input});

    vector<float> input_data{-1, 3};
    vector<float> expected_out{-1.1113307, 3.152103};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(rt_shape, input_data);
    test_case.add_expected_output(rt_shape, expected_out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, selu_4Dfprop)
{
    Shape in_shape{4};
    Shape c_shape{1};
    element::Type et = element::f32;

    auto input = make_shared<op::Parameter>(et, in_shape);
    auto alpha = op::Constant::create(et, c_shape, {1.67326324});
    auto lambda = op::Constant::create(et, c_shape, {1.05070098});
    auto selu = make_shared<op::v0::Selu>(input, alpha, lambda);
    auto f = make_shared<Function>(selu, ParameterVector{input});

    vector<float> in_vec{-1.0, 0.0, 1.0, 2.0};
    vector<float> out_vec{-1.1113307, 0., 1.050701, 2.101402};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_shape, in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run_with_tolerance_as_fp(1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, selu_1Dfprop)
{
    Shape in_shape{1};
    Shape c_shape{1};
    element::Type et = element::f32;

    auto input = make_shared<op::Parameter>(et, in_shape);
    auto alpha = op::Constant::create(et, c_shape, {1.67326324});
    auto lambda = op::Constant::create(et, c_shape, {1.05070098});
    auto selu = make_shared<op::v0::Selu>(input, alpha, lambda);
    auto f = make_shared<Function>(selu, ParameterVector{input});

    vector<float> in_vec{112.0};
    vector<float> out_vec{117.67851};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_shape, in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run_with_tolerance_as_fp(1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, selu_3Dfprop_negative)
{
    Shape in_shape{3};
    Shape c_shape{1};
    element::Type et = element::f32;

    auto input = make_shared<op::Parameter>(et, in_shape);
    auto alpha = op::Constant::create(et, c_shape, {1.67326324});
    auto lambda = op::Constant::create(et, c_shape, {1.05070098});
    auto selu = make_shared<op::v0::Selu>(input, alpha, lambda);
    auto f = make_shared<Function>(selu, ParameterVector{input});

    vector<float> in_vec{-3.0, -12.5, -7.0};
    vector<float> out_vec{-1.6705687, -1.7580928, -1.7564961};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_shape, in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run_with_tolerance_as_fp(1e-4f);
}
