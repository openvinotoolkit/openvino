// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Floor>(),
                                         {-2.5f, -2.0f, 0.3f, 4.8f},
                                         {-3.0f, -2.0f, 0.0f, 4.0f},
                                         Shape{2, 2});
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int32)
{
    test_unary<TestEngine, element::i32>(unary_func<op::Floor>(),
                                         {-2, -136314888, 0x40000010, 0x40000001},
                                         {-2, -136314888, 0x40000010, 0x40000001},
                                         Shape{2, 2});
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int64)
{
    // This tests large numbers that will not fit in a double
    test_unary<TestEngine, element::i64>(
        unary_func<op::Floor>(), {0, 1, 0x4000000000000001}, {0, 1, 0x4000000000000001});
}
