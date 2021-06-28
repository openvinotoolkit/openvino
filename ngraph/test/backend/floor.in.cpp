// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    test::make_unary_test<TestEngine, op::Floor, element::f32>(Shape{2, 2})
        .test({-2.5f, -2.0f, 0.3f, 4.8f}, {-3.0f, -2.0f, 0.0f, 4.0f});
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int32)
{
    test::make_unary_test<TestEngine, op::Floor, element::i32>(Shape{2, 2})
        .test({-2, -136314888, 0x40000010, 0x40000001}, {-2, -136314888, 0x40000010, 0x40000001});
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int64)
{
    // This tests large numbers that will not fit in a double
    test::make_unary_test<TestEngine, op::Floor, element::i64>(Shape{3}).test(
        {0, 1, 0x4000000000000001}, {0, 1, 0x4000000000000001});
}
