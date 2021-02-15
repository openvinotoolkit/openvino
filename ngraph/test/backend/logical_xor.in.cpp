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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, logical_xor)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Xor>(A, B), ParameterVector{A, B});

    std::vector<char> a{1, 0, 1, 1, 1, 0, 1, 0};
    std::vector<char> b{0, 0, 1, 0, 0, 1, 1, 0};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<char>({a, b});
    test_case.add_expected_output<char>(shape, {1, 0, 0, 1, 1, 1, 0, 0});
    test_case.run();
}
