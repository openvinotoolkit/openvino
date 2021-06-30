// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename OpType, element::Type_t et>
test::unary_test<TestEngine, et, element::Type_t::boolean>
    comparison_test(const ngraph::PartialShape& shape)
{
    auto A = make_shared<op::Parameter>(et, shape);
    auto B = make_shared<op::Parameter>(et, shape);
    auto f = make_shared<Function>(make_shared<OpType>(A, B), ParameterVector{A, B});

    return test::unary_test<TestEngine, et, element::Type_t::boolean>(f);
}

NGRAPH_TEST(${BACKEND_NAME}, equal)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::Equal, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0, 1, 1}, {1, 8, 4, 8, 0, 0, 1, 1.5}}, {1, 1, 0, 0, 0, 1, 1, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, notequal)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::NotEqual, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0, 1, 1}, {1, 8, 4, 8, 0, 0, 1, 1.5}}, {0, 0, 1, 1, 1, 0, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, greater)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::Greater, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0.5, 2, 1}, {1, 2, 4, 8, 0, 0, 1, 1.5}}, {0, 1, 0, 1, 0, 1, 1, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, greater_int64)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::Greater, element::i64>(shape).test(
        {{0x4000000000000002, 0x4000000000000006, -8, 17, -5, 5, 2, 1},
         {0x4000000000000001, 0x4000000000000002, 4, 8, 0, 0, 1, 2}},
        {1, 1, 0, 1, 0, 1, 1, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, greatereq)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::GreaterEqual, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0, 2, 1}, {1, 2, -8, 8, 0, 0, 0.5, 1.5}}, {1, 1, 1, 1, 0, 1, 1, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, less)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::Less, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0.5, 2, 1}, {1, 2, 4, 8, 0, 0, 1, 1.5}}, {0, 0, 1, 0, 1, 0, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::LessEqual, element::f32>(shape).test(
        {{1, 8, -8, 17, -0.5, 0, 2, 1}, {1, 2, -8, 8, 0, 0, 0.5, 1.5}}, {1, 0, 1, 0, 1, 1, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_int32)
{
    Shape shape{2, 2};

    comparison_test<op::v1::LessEqual, element::i32>(shape).test(
        {{0x40000170, 0x40000005, 0x40000005, -5}, {0x40000140, 0x40000001, 0x40000005, 0}},
        {0, 0, 1, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_bool)
{
    Shape shape{2, 2, 2};

    comparison_test<op::v1::LessEqual, element::boolean>(shape).test(
        {{1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0}}, {0, 0, 0, 0, 0, 0, 0, 0});
}
