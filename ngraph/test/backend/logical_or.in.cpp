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
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, logical_or)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Or>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 1, 1, 1, 1, 0}), read_vector<char>(result));
}
