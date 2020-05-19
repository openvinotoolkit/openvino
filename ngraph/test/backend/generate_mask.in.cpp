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

NGRAPH_TEST(${BACKEND_NAME}, generate_mask)
{
    Shape scalar{};
    Shape result_shape{1, 128};
    const unsigned int seed = 777;
    auto training = op::Constant::create(element::f32, Shape{}, {1});
    auto gen_mask =
        make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5, false);
    auto gen_mask2 =
        make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5, false);
    auto f = make_shared<Function>(NodeVector{gen_mask, gen_mask2}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto is_not_zero_or_one = [](float num) { return num != 0.f && num != 1.f; };

    auto result_tv1 = backend->create_tensor<float>(result_shape);
    auto result_tv2 = backend->create_tensor<float>(result_shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result_tv1, result_tv2}, {});
    auto result1 = read_vector<float>(result_tv1);
    auto result2 = read_vector<float>(result_tv2);
    ASSERT_TRUE(test::all_close_f(result1, result2));
    ASSERT_FALSE(std::any_of(result1.begin(), result1.end(), is_not_zero_or_one));
    handle->call_with_validate({result_tv1, result_tv2}, {});
    auto result1_2 = read_vector<float>(result_tv1);
    auto result2_2 = read_vector<float>(result_tv2);
    ASSERT_FALSE(test::all_close_f(result1, result1_2));
    ASSERT_FALSE(std::any_of(result1_2.begin(), result1_2.end(), is_not_zero_or_one));
    ASSERT_FALSE(test::all_close_f(result2, result2_2));
    ASSERT_FALSE(std::any_of(result2_2.begin(), result2_2.end(), is_not_zero_or_one));
}

NGRAPH_TEST(${BACKEND_NAME}, generate_mask2)
{
    Shape scalar{};
    Shape result_shape{1, 128};
    const unsigned int seed = 777;
    auto training = op::Constant::create(element::f32, Shape{}, {1});
    auto gen_mask =
        make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5, true);
    auto gen_mask2 =
        make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5, true);
    auto f = make_shared<Function>(NodeVector{gen_mask, gen_mask2}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto is_not_zero_or_one = [](float num) { return num != 0.f && num != 1.f; };

    auto result_tv1 = backend->create_tensor<float>(result_shape);
    auto result_tv2 = backend->create_tensor<float>(result_shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result_tv1, result_tv2}, {});
    auto result1 = read_vector<float>(result_tv1);
    auto result2 = read_vector<float>(result_tv2);
    ASSERT_TRUE(test::all_close_f(result1, result2));
    ASSERT_FALSE(std::any_of(result1.begin(), result1.end(), is_not_zero_or_one));

    auto result_tv1_2 = backend->create_tensor<float>(result_shape);
    auto result_tv2_2 = backend->create_tensor<float>(result_shape);
    handle->call_with_validate({result_tv1_2, result_tv2_2}, {});
    auto result1_2 = read_vector<float>(result_tv1_2);
    auto result2_2 = read_vector<float>(result_tv2_2);
    ASSERT_TRUE(test::all_close_f(result1, result1_2));
    ASSERT_FALSE(std::any_of(result1_2.begin(), result1_2.end(), is_not_zero_or_one));
    ASSERT_TRUE(test::all_close_f(result2, result2_2));
    ASSERT_FALSE(std::any_of(result2_2.begin(), result2_2.end(), is_not_zero_or_one));
}

NGRAPH_TEST(${BACKEND_NAME}, dyn_generate_mask)
{
    const unsigned int seed = 777;
    auto training = op::Constant::create(element::f32, Shape{}, {1});
    auto result_shape =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto gen_mask =
        make_shared<op::v1::GenerateMask>(training, result_shape, element::f32, seed, 0.5, true);
    auto gen_mask2 =
        make_shared<op::v1::GenerateMask>(training, result_shape, element::f32, seed, 0.5, true);
    auto f = make_shared<Function>(NodeVector{gen_mask, gen_mask2}, ParameterVector{result_shape});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto is_not_zero_or_one = [](float num) { return num != 0.f && num != 1.f; };

    vector<int64_t> shapes = {1, 128};
    auto shape_result = backend->create_tensor(element::i64, Shape{shapes.size()});
    copy_data(shape_result, shapes);
    auto result_tv1 = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto result_tv2 = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto handle = backend->compile(f);
    handle->call_with_validate({result_tv1, result_tv2}, {shape_result});
    ASSERT_EQ(result_tv1->get_shape(), (Shape{1, 128}));
    ASSERT_EQ(result_tv2->get_shape(), (Shape{1, 128}));
    auto result1 = read_vector<float>(result_tv1);
    auto result2 = read_vector<float>(result_tv2);
    ASSERT_TRUE(test::all_close_f(result1, result2));
    ASSERT_FALSE(std::any_of(result1.begin(), result1.end(), is_not_zero_or_one));

    auto result_tv1_2 = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto result_tv2_2 = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    handle->call_with_validate({result_tv1_2, result_tv2_2}, {shape_result});
    auto result1_2 = read_vector<float>(result_tv1_2);
    auto result2_2 = read_vector<float>(result_tv2_2);
    ASSERT_TRUE(test::all_close_f(result1, result1_2));
    ASSERT_FALSE(std::any_of(result1_2.begin(), result1_2.end(), is_not_zero_or_one));
    ASSERT_TRUE(test::all_close_f(result2, result2_2));
    ASSERT_FALSE(std::any_of(result2_2.begin(), result2_2.end(), is_not_zero_or_one));
}
