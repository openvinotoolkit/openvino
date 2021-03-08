//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false, false});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false, false});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false, false});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false, false});
    test_case.run();
}
