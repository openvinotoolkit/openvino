// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, exp)
{
    test::make_unary_test<TestEngine, op::Exp, element::f32>(Shape{8}).test(
        {-4, -3, -2, -1, 0, 1, 2, 3}, std::exp);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_negative)
{
    test::make_unary_test<TestEngine, op::Exp, element::f32>(Shape{5}).test({-4, -3, -2, -1, -5},
                                                                            std::exp);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_scalar)
{
    test::make_unary_test<TestEngine, op::Exp, element::f32>(Shape{}).test(
        {13}, std::exp, DEFAULT_FLOAT_TOLERANCE_BITS + 2);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_in_place)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto T = make_shared<op::Exp>(A);
    auto T2 = make_shared<op::Exp>(T);
    auto f = make_shared<Function>(T2, ParameterVector{A});

    test::unary_test<TestEngine, element::f32>(f).test(
        {1, 3}, [](float x) { return std::exp(std::exp(x)); }, DEFAULT_FLOAT_TOLERANCE_BITS + 2);
}