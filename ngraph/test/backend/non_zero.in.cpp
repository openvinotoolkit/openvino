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
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, non_zero)
{
    PartialShape p_shape = PartialShape::dynamic();
    auto p = make_shared<op::Parameter>(element::Type_t::f32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::Type_t::i32);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto cfun = backend->compile(fun);

    auto input = backend->create_tensor(element::Type_t::f32, Shape{3, 2});
    copy_data(input, vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f});

    std::vector<int32_t> expected_result{2, 2, 0, 1};
    Shape expected_output_shape{2, 2};

    auto result = make_shared<HostTensor>();
    cfun->call_with_validate({result}, {input});

    EXPECT_EQ(result->get_element_type(), element::Type_t::i32);
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    auto result_data = read_vector<int32_t>(result);
    ASSERT_EQ(result_data, expected_result);
}

NGRAPH_TEST(${BACKEND_NAME}, non_zero_all_1s)
{
    PartialShape p_shape = PartialShape::dynamic();
    auto p = make_shared<op::Parameter>(element::Type_t::i32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::Type_t::i64);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto cfun = backend->compile(fun);

    Shape input_shape{3, 2};
    vector<int32_t> input_data(shape_size(input_shape), 1);
    auto input = backend->create_tensor(element::Type_t::i32, input_shape);
    copy_data(input, input_data);

    std::vector<int64_t> expected_result{0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1};
    Shape expected_output_shape{2, 6};

    auto result = make_shared<HostTensor>();
    cfun->call_with_validate({result}, {input});

    EXPECT_EQ(result->get_element_type(), element::Type_t::i64);
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    auto result_data = read_vector<int64_t>(result);
    ASSERT_EQ(result_data, expected_result);
}

NGRAPH_TEST(${BACKEND_NAME}, non_zero_all_0s)
{
    PartialShape p_shape = PartialShape::dynamic();
    auto p = make_shared<op::Parameter>(element::Type_t::i32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::Type_t::i64);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto cfun = backend->compile(fun);

    Shape input_shape{3, 2};
    vector<int32_t> input_data(shape_size(input_shape), 0);
    auto input = backend->create_tensor(element::Type_t::i32, input_shape);
    copy_data(input, input_data);

    Shape expected_output_shape{input_shape.size(), 0};

    auto result = make_shared<HostTensor>();
    cfun->call_with_validate({result}, {input});

    EXPECT_EQ(result->get_element_type(), element::Type_t::i64);
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    auto result_data = read_vector<int64_t>(result);
    ASSERT_EQ(result_data.data(), nullptr);
}
