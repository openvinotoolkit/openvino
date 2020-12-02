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

NGRAPH_TEST(${BACKEND_NAME}, logical_and)
{
    Shape shape{3, 4};
    auto A = make_shared<op::Parameter>(element::Type_t::boolean, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::boolean, shape);
    auto f =
        make_shared<Function>(std::make_shared<op::v1::LogicalAnd>(A, B), ParameterVector{A, B});

    std::vector<bool> a{true, true, true, true, true, false, true, false, false, true, true, true};
    std::vector<bool> b{true, true, true, true, true, false, true, false, false, true, true, false};

    auto test_case_1 = test::TestCase<TestEngine>(f);
    test_case_1.add_multiple_inputs<bool>({a, b});
    test_case_1.add_expected_output<float>(shape, {1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 0.});
    test_case_1.run();
}
