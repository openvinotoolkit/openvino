// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <ngraph::element::Type_t et>
test::unary_test<TestEngine, et> swish_test(const ngraph::PartialShape& in_shape)
{
    auto args0 = make_shared<op::Parameter>(et, in_shape);
    auto args1 = make_shared<op::Parameter>(et, Shape{});
    auto swish = make_shared<op::v4::Swish>(args0, args1);
    auto f = make_shared<Function>(swish, ParameterVector{args0, args1});
    return test::unary_test<TestEngine, et>(f);
}

template <typename T>
T f_swish(T x, T beta)
{
    return x / (1.0f + std::exp(x * beta * -1.0f));
}

NGRAPH_TEST(${BACKEND_NAME}, swish_2D_with_beta0_6)
{
    Shape in_shape{2, 4};
    auto beta = 0.6f;

    swish_test<element::f32>(in_shape).test({{0.4, -5.7, -6, 3, -0.9, 23, 5, 3.3}, {beta}},
                                            [beta](float x) { return f_swish(x, beta); });
}

NGRAPH_TEST(${BACKEND_NAME}, swish_2D_without_beta)
{
    Shape in_shape{2, 3};

    test::make_unary_test<TestEngine, op::v4::Swish, element::f32>(in_shape).test(
        {1, 8, -8, 17, -0.5, -1}, [](float x) { return f_swish(x, 1.0f); });
}

NGRAPH_TEST(${BACKEND_NAME}, swish_4D_with_beta0_33)
{
    Shape in_shape{2, 2, 1, 2};
    auto beta = 0.33f;

    swish_test<element::f32>(in_shape).test({{0.1, 0.6, 20, -7, -5.3, 3.5, -9, 11}, {beta}},
                                            [beta](float x) { return f_swish(x, beta); });
}
