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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto C = make_shared<op::v1::Add>(A, B);
    auto D = make_shared<op::v1::Multiply>(A, B);
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, ParameterVector{A, B});

    vector<float> a{0, 1, 2, 3};
    vector<float> b{1, 2, 3, 4};
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedD);
    test_case.add_expected_output<float>(shape, expectedD);
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedE);
    test_case.add_expected_output<float>(shape, expectedE);
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}
