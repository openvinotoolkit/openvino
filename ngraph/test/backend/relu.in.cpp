// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    test::make_unary_test<TestEngine, op::Relu, element::f32>(Shape{2, 5})
        .test({1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5},
              {1, 8, 0, 17, 0, 1, 8, 0, 17, 0},
              MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop_i32)
{
    test::make_unary_test<TestEngine, op::Relu, element::i32>(Shape{2, 5})
        .test({1, 8, -8, 17, -2, 1, 8, -8, 17, -1}, {1, 8, 0, 17, 0, 1, 8, 0, 17, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    test::make_unary_test<TestEngine, op::Relu, element::f32>(Shape{2, 2, 2, 2})
        .test({1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1},
              {1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1},
              MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::v1::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, ParameterVector{B});

    test::unary_test<TestEngine, element::f32>(f).test(
        {{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5}, shape_a},
        {{1, 8, 0, 17, 0, 1, 8, 0, 17, 0}, shape_rt},
        MIN_FLOAT_TOLERANCE_BITS);
}
