// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, round_half_to_even)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::v5::Round>(op::v5::Round::RoundMode::HALF_TO_EVEN),
        {0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
        {1.0f, 2.0f, 2.0f, 2.0f, -4.0f},
        MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, round_away_from_zero)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::v5::Round>(op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO),
        {0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
        {1.0f, 3.0f, 2.0f, 2.0f, -5.0f},
        MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, round_2D)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::v5::Round>(op::v5::Round::RoundMode::HALF_TO_EVEN),
        {0.1f,
         0.5f,
         0.9f,
         1.2f,
         1.5f,
         1.8f,
         2.3f,
         2.5f,
         2.7f,
         -1.1f,
         -1.5f,
         -1.9f,
         -2.2f,
         -2.5f,
         -2.8f},
        {0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f},
        Shape{3, 5},
        Shape{3, 5},
        MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, round_int64)
{
    // This tests large numbers that will not fit in a double
    test_unary<TestEngine, element::i64>(
        unary_func<op::v5::Round>(op::v5::Round::RoundMode::HALF_TO_EVEN),
        {0, 1, 0x4000000000000001},
        {0, 1, 0x4000000000000001});
}
