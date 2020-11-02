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

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), ParameterVector{A});

    std::vector<float> a{1, -2, 0, -4.75f, 8.75f, -8.75f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(a);
    test_case.add_expected_output<float>(shape, {-1, 2, 0, 4.75f, -8.75f, 8.75f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, negative_i32)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto relu = make_shared<op::Negative>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    std::vector<int32_t> a{1, 8, -8, 17, -2, 1, 8, -8, 17, -1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(shape_a, a);
    test_case.add_expected_output<int32_t>(shape_rt, {-1, -8, 8, -17, 2, -1, -8, 8, -17, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, negative_f32)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Negative>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    std::vector<float> a{1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape_a, a);
    test_case.add_expected_output<float>(
        shape_rt, {-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f});
    test_case.run();
}
