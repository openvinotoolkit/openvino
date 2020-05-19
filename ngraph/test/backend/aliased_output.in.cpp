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
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    auto D = A * B;
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out1 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out2 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out3 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out4 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out5 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out6 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out7 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    auto handle = backend->compile(f);
    handle->call_with_validate({out1, out2, out3, out4, out5, out6, out7}, {a, b});
    EXPECT_TRUE(test::all_close_f(expectedC, read_vector<float>(out1), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedC, read_vector<float>(out2), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedD, read_vector<float>(out3), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedD, read_vector<float>(out4), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedC, read_vector<float>(out5), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedE, read_vector<float>(out6), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(expectedE, read_vector<float>(out7), MIN_FLOAT_TOLERANCE_BITS));
}
