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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, multiple_backends)
{
    Shape shape{2, 2};
    auto A1 = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B1 = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(A1 + B1, ParameterVector{A1, B1});

    auto A2 = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B2 = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto g = make_shared<Function>(A2 * B2, ParameterVector{A2, B2});

    auto backend1 = runtime::Backend::create("${BACKEND_NAME}");

    auto backend2 = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a1 = backend1->create_tensor(element::Type_t::f32, shape);
    shared_ptr<runtime::Tensor> b1 = backend1->create_tensor(element::Type_t::f32, shape);
    shared_ptr<runtime::Tensor> result1 = backend1->create_tensor(element::Type_t::f32, shape);

    shared_ptr<runtime::Tensor> a2 = backend2->create_tensor(element::Type_t::f32, shape);
    shared_ptr<runtime::Tensor> b2 = backend2->create_tensor(element::Type_t::f32, shape);
    shared_ptr<runtime::Tensor> result2 = backend2->create_tensor(element::Type_t::f32, shape);

    copy_data(a1, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b1, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    copy_data(a2, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b2, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle1 = backend1->compile(f);
    handle1->call_with_validate({result1}, {a1, b1});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result1),
                                  (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector()));

    auto handle2 = backend2->compile(g);
    handle2->call_with_validate({result2}, {a2, b2});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result2),
                                  (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector()));
}
