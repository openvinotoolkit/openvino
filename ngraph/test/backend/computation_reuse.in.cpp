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

NGRAPH_TEST(${BACKEND_NAME}, computation_reuse)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b, true);
    Shape shape_r{1, 32, 2, 2};
    auto conv = make_shared<op::Convolution>(A,
                                             B,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool = make_shared<op::AvgPool>(conv, pool_shape);
    auto bias = make_shared<op::Broadcast>(
        op::Constant::create(element::f32, Shape{}, {2.14}), shape_r, AxisSet{0, 1, 2, 3});
    auto result_op = make_shared<op::Result>(pool + bias);
    auto f = make_shared<Function>(ResultVector{result_op}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> input(64, 1.0f);
    vector<float> weights(512, 0.5f);
    vector<float> rv(128);

    auto a = backend->create_tensor(element::f32, shape_a);
    auto b = backend->create_tensor(element::f32, shape_b);
    auto result = backend->create_tensor(element::f32, shape_r);

    copy_data(a, input);
    copy_data(b, weights);

    auto exec = backend->compile(f);
    exec->call_with_validate({result}, {a, b});

    vector<float> rv_saved(read_vector<float>(result));

    b->set_stale(false);
    exec->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(rv_saved, read_vector<float>(result)));
}
