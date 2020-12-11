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

NGRAPH_TEST(${BACKEND_NAME}, evaluate_dynamic_gather_elements)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<int32_t> data{1, 2, 3, 4};
    std::vector<int32_t> indices{0, 1, 0, 0};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{1, 4, 1, 2});
    test_case.run();
}
