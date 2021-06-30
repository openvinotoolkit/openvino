// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <ngraph::element::Type_t et>
test::unary_test<TestEngine, et> add_test(const ngraph::PartialShape& shapeA,
                                          const ngraph::PartialShape& shapeB)
{
    auto A = make_shared<op::Parameter>(et, shapeA);
    auto B = make_shared<op::Parameter>(et, shapeB);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    return test::unary_test<TestEngine, et>(f);
}

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    add_test<element::f32>(shape, shape).test({{1, 2, 3, 4}, {5, 6, 7, 8}}, {6, 8, 10, 12});
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    add_test<element::f32>(shape, shape).test({{1, 2, 3, 4}, {5, 6, 7, 8}}, {6, 8, 10, 12});
}

NGRAPH_TEST(${BACKEND_NAME}, add_in_place)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto T = make_shared<op::v1::Add>(A, B);
    auto T2 = make_shared<op::v1::Add>(T, T);
    auto T3 = make_shared<op::v1::Add>(T2, T2);
    auto T4 = make_shared<op::v1::Add>(T3, T3);

    auto f = make_shared<Function>(T4, ParameterVector{A, B});

    test::unary_test<TestEngine, element::f32>(f).test({{1, 2, 3, 4}, {5, 6, 7, 8}},
                                                       {48, 64, 80, 96});
}

NGRAPH_TEST(${BACKEND_NAME}, add_broadcast)
{
    Shape shape_a{1, 2};
    Shape shape_b{3, 2, 2};

    add_test<element::f32>(shape_a, shape_b)
        .test({{1, 2}, {5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3}},
              {6, 8, 8, 10, 3, 5, 2, 7, 7, 9, 2, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, add_scalars)
{
    Shape shape{};
    add_test<element::f32>(shape, shape).test({{2}, {8}}, {10});
}

NGRAPH_TEST(${BACKEND_NAME}, add_vector_and_scalar)
{
    Shape shape_a{2, 2};
    Shape shape_b{};

    add_test<element::f32>(shape_a, shape_b).test({{2, 4, 7, 8}, {8}}, {10, 12, 15, 16});
}
