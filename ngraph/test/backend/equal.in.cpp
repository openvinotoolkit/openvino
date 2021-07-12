// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, equal_f32_no_broadcast)
{
    Shape shape{4};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, equal_f32_broadcast)
{
    Shape shape1{2, 1, 2, 1};
    Shape shape2{1, 2, 1};
    Shape shape_output{2, 1, 2, 1};
    auto A = op::Constant::create(element::f32, shape1, {2.0f, 1.0f, 4.0f, 1.0f});
    auto B = op::Constant::create(element::f32, shape2, {1.0f, 1.0f});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape_output, {false, true, false, true});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, equal_f32_negative)
{
    Shape shape{3};
    auto A = op::Constant::create(element::f32, shape, {-1.5f, -2.5f, -5.0f});
    auto B = op::Constant::create(element::f32, shape, {1.5f, -4.5f, -5.0f});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, equal_i32)
{
    Shape shape{3};
    auto A = op::Constant::create(element::i32, shape, {1, 2, 3});
    auto B = op::Constant::create(element::i32, shape, {4, 2, 1});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<uint8_t>(shape, {0, 1, 0});
    test_case.run();
}