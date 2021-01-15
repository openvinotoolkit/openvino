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
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// This tests a backend's implementation of the two parameter version of create_tensor
NGRAPH_TEST(${BACKEND_NAME}, create_tensor_1)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> av = {1, 2, 3, 4};
    vector<float> bv = {5, 6, 7, 8};
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    copy_data(a, av);
    copy_data(b, bv);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected = {6, 8, 10, 12};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, get_parameters_and_results)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto arg = make_shared<op::v1::Multiply>(make_shared<op::v1::Add>(A, B), C);
    auto f = make_shared<Function>(arg, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    auto parameters = handle->get_parameters();
    auto results = handle->get_results();
    ASSERT_EQ(parameters.size(), 3);
    ASSERT_EQ(results.size(), 1);

    // This part can't be enabled until we force backends to make a copy of the source graph
    // auto func_parameters = f->get_parameters();
    // auto func_results = f->get_results();
    // for (size_t i = 0; i < 3; ++i)
    // {
    //     EXPECT_NE(parameters[i], func_parameters[i]);
    // }
    // for (size_t i = 0; i < 1; ++i)
    // {
    //     EXPECT_NE(results[i], func_results[i]);
    // }
}
