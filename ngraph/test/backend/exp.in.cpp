// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, exp)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Exp>(), {-4, -3, -2, -1, 0, 1, 2, 3}, std::exp);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_negative)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Exp>(), {-4, -3, -2, -1, -5}, std::exp);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_scalar)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Exp>(), {13}, std::exp, {}, DEFAULT_FLOAT_TOLERANCE_BITS + 2);
}

template <typename T>
static T exp_exp(T x)
{
    return std::exp(std::exp(x));
}

NGRAPH_TEST(${BACKEND_NAME}, exp_in_place)
{
    auto creator = [](const ngraph::element::Type& ele_type, const PartialShape& pshape) {
        auto A = make_shared<op::Parameter>(ele_type, pshape);
        auto T = make_shared<op::Exp>(A);
        auto T2 = make_shared<op::Exp>(T);

        return make_shared<Function>(T2, ParameterVector{A});
    };

    test_unary<TestEngine, element::f32>(
        creator, {1, 3}, exp_exp, {}, DEFAULT_FLOAT_TOLERANCE_BITS + 2);
}