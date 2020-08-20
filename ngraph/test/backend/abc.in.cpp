//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{5, 6, 7, 8};
    std::vector<float> c{9, 10, 11, 12};

    // (a + b) * c
    auto test_case_1 = test::TestCase<TestEngine>(f);
    test_case_1.add_multiple_inputs<float>({a, b, c});
    test_case_1.add_expected_output<float>(shape, {54, 80, 110, 144});
    test_case_1.run();

    // (b + a) * c
    auto test_case_2 = test::TestCase<TestEngine>(f);
    test_case_2.add_multiple_inputs<float>({b, a, c});
    test_case_2.add_expected_output<float>(shape, {54, 80, 110, 144});
    test_case_2.run();

    // (a + c) * b
    auto test_case_3 = test::TestCase<TestEngine>(f);
    test_case_3.add_multiple_inputs<float>({a, c, b});
    test_case_3.add_expected_output<float>(shape, {50, 72, 98, 128});
    test_case_3.run();
}

NGRAPH_TEST(${BACKEND_NAME}, abc_int64)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto C = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    std::vector<int64_t> a{1, 2, 3, 4};
    std::vector<int64_t> b{5, 6, 7, 8};
    std::vector<int64_t> c{9, 10, 11, 12};

    // (a + b) * c
    auto test_case_1 = test::TestCase<TestEngine>(f);
    test_case_1.add_multiple_inputs<int64_t>({a, b, c});
    test_case_1.add_expected_output<int64_t>(shape, {54, 80, 110, 144});
    test_case_1.run();

    // (b + a) * c
    auto test_case_2 = test::TestCase<TestEngine>(f);
    test_case_2.add_multiple_inputs<int64_t>({b, a, c});
    test_case_2.add_expected_output<int64_t>(shape, {54, 80, 110, 144});
    test_case_2.run();

    // (a + c) * b
    auto test_case_3 = test::TestCase<TestEngine>(f);
    test_case_3.add_multiple_inputs<int64_t>({a, c, b});
    test_case_3.add_expected_output<int64_t>(shape, {50, 72, 98, 128});
    test_case_3.run();
}
