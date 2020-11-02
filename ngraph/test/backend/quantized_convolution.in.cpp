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

NGRAPH_TEST(${BACKEND_NAME}, quantized_conv_int32_output)
{
    Shape shape_a{1, 1, 3, 4};
    Shape shape_b{1, 1, 3, 3};
    Shape shape_r{1, 1, 3, 4};
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<uint8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::u8, shape_b);
    auto C = make_shared<op::Parameter>(element::f32, Shape{});
    auto D = op::Constant::create(element::u8, Shape{}, {0});
    auto E = make_shared<op::Parameter>(element::f32, Shape{});
    auto F = op::Constant::create(element::u8, Shape{}, {0});
    auto G = make_shared<op::Parameter>(element::f32, Shape{});
    auto H = op::Constant::create(element::i32, Shape{}, {0});
    auto CV = make_shared<op::QuantizedConvolution>(A,
                                                    B,
                                                    Strides{1, 1},
                                                    Strides{1, 1},
                                                    CoordinateDiff{1, 1},
                                                    CoordinateDiff{1, 1},
                                                    Strides{1, 1},
                                                    C,
                                                    D,
                                                    E,
                                                    F,
                                                    G,
                                                    H,
                                                    element::i32);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, C, E, G});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::u8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, Shape{});
    copy_data(c, vector<float>{1.0f});
    auto d = backend->create_tensor(element::f32, Shape{});
    copy_data(d, vector<float>{1.0f});
    auto e = backend->create_tensor(element::f32, Shape{});
    copy_data(e, vector<float>{1.0f});
    auto result = backend->create_tensor(element::i32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c, d, e});
    EXPECT_EQ((vector<int32_t>{22, 34, 30, 32, 38, 72, 90, 43, 33, 52, 43, 39}),
              read_vector<int32_t>(result));
}
