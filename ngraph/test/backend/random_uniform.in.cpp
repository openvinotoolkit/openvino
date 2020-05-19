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

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_all_static_seed_unused)
{
    auto min_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{63.0f});
    auto max_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{120.0f});
    auto result_shape =
        make_shared<op::Constant>(element::i64, Shape{3}, std::vector<float>{50, 200, 100});
    auto use_fixed_seed =
        make_shared<op::Constant>(element::boolean, Shape{}, std::vector<char>{0});
    size_t fixed_seed = 9999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    auto f = make_shared<Function>(NodeVector{ru}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto ex = backend->compile(f);

    auto t_r = backend->create_tensor(element::f32, Shape{50, 200, 100});

    ex->call_with_validate({t_r}, {});
    auto results_0 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x <= 120.0f); }));

    ex->call_with_validate({t_r}, {});
    auto results_1 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x <= 120.0f); }));

    ASSERT_FALSE(test::all_close_f(results_0, results_1)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_all_static_seed_used)
{
    auto min_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{63.0f});
    auto max_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{120.0f});
    auto result_shape =
        make_shared<op::Constant>(element::i64, Shape{3}, std::vector<float>{50, 200, 100});
    auto use_fixed_seed =
        make_shared<op::Constant>(element::boolean, Shape{}, std::vector<char>{1});
    size_t fixed_seed = 9999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    auto f = make_shared<Function>(NodeVector{ru}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto ex = backend->compile(f);

    auto t_r = backend->create_tensor(element::f32, Shape{50, 200, 100});

    ex->call_with_validate({t_r}, {});
    auto results_0 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x <= 120.0f); }));

    ex->call_with_validate({t_r}, {});
    auto results_1 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x <= 120.0f); }));

    ASSERT_TRUE(test::all_close_f(results_0, results_1));
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_seed_use_dynamic)
{
    auto min_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{63.0f});
    auto max_val = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{120.0f});
    auto result_shape =
        make_shared<op::Constant>(element::i64, Shape{3}, std::vector<float>{50, 200, 100});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 9999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    auto f = make_shared<Function>(NodeVector{ru}, ParameterVector{use_fixed_seed});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto ex = backend->compile(f);

    auto t_use_fixed_seed = backend->create_tensor(element::boolean, Shape{});
    auto t_r = backend->create_tensor(element::f32, Shape{50, 200, 100});

    copy_data(t_use_fixed_seed, std::vector<char>{1});

    ex->call_with_validate({t_r}, {t_use_fixed_seed});
    auto results_0 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x <= 120.0f); }));

    ex->call_with_validate({t_r}, {t_use_fixed_seed});
    auto results_1 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x <= 120.0f); }));

    ASSERT_TRUE(test::all_close_f(results_0, results_1));

    copy_data(t_use_fixed_seed, std::vector<char>{0});

    ex->call_with_validate({t_r}, {t_use_fixed_seed});
    auto results_2 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x <= 120.0f); }));

    ex->call_with_validate({t_r}, {t_use_fixed_seed});
    auto results_3 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x >= 63.0f); }));
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x <= 120.0f); }));

    ASSERT_FALSE(test::all_close_f(results_2, results_3)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_all_static_range_dynamic)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape =
        make_shared<op::Constant>(element::i64, Shape{3}, std::vector<float>{50, 200, 100});
    auto use_fixed_seed =
        make_shared<op::Constant>(element::boolean, Shape{}, std::vector<char>{0});
    size_t fixed_seed = 9999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    auto f = make_shared<Function>(NodeVector{ru}, ParameterVector{min_val, max_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto ex = backend->compile(f);

    auto t_min_val = backend->create_tensor(element::f32, Shape{});
    auto t_max_val = backend->create_tensor(element::f32, Shape{});
    auto t_r = backend->create_tensor(element::f32, Shape{50, 200, 100});

    copy_data(t_min_val, std::vector<float>{-10.0f});
    copy_data(t_max_val, std::vector<float>{10.0f});

    ex->call_with_validate({t_r}, {t_min_val, t_max_val});
    auto results_0 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x <= 10.0f); }));

    ex->call_with_validate({t_r}, {t_min_val, t_max_val});
    auto results_1 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x <= 10.0f); }));

    ASSERT_FALSE(test::all_close_f(results_0, results_1)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";

    copy_data(t_min_val, std::vector<float>{23.0f});
    copy_data(t_max_val, std::vector<float>{490.0f});

    ex->call_with_validate({t_r}, {t_min_val, t_max_val});
    auto results_2 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x >= 23.0f); }));
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x <= 490.0f); }));

    ex->call_with_validate({t_r}, {t_min_val, t_max_val});
    auto results_3 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x >= 23.0f); }));
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x <= 490.0f); }));

    ASSERT_FALSE(test::all_close_f(results_2, results_3)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_dynamic_shapes)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 9999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    auto f = make_shared<Function>(NodeVector{ru},
                                   ParameterVector{min_val, max_val, result_shape, use_fixed_seed});

    // Getting a dynamic backend here.
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    shared_ptr<runtime::Tensor> t_min_val;
    shared_ptr<runtime::Tensor> t_max_val;
    shared_ptr<runtime::Tensor> t_result_shape;
    shared_ptr<runtime::Tensor> t_use_fixed_seed;
    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    // Set some starting values.
    t_min_val = backend->create_tensor(element::f32, Shape{});
    t_max_val = backend->create_tensor(element::f32, Shape{});
    t_result_shape = backend->create_tensor(element::i64, Shape{3});
    t_use_fixed_seed = backend->create_tensor(element::boolean, Shape{});
    copy_data(t_min_val, std::vector<float>{-10.0f});
    copy_data(t_max_val, std::vector<float>{10.0f});
    copy_data(t_result_shape, std::vector<int64_t>{10, 100, 100});
    copy_data(t_use_fixed_seed, std::vector<char>{0});

    ex->call_with_validate({t_r}, {t_min_val, t_max_val, t_result_shape, t_use_fixed_seed});
    ASSERT_EQ(t_r->get_shape(), (Shape{10, 100, 100}));

    auto results_0 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_0.begin(), results_0.end(), [](float x) { return (x <= 10.0f); }));

    ex->call_with_validate({t_r}, {t_min_val, t_max_val, t_result_shape, t_use_fixed_seed});
    auto results_1 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_1.begin(), results_1.end(), [](float x) { return (x <= 10.0f); }));

    ASSERT_FALSE(test::all_close_f(results_0, results_1)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";

    // Change the shape, run again with same executable.
    t_result_shape = backend->create_tensor(element::i64, Shape{2});
    copy_data(t_result_shape, std::vector<int64_t>{500, 2000});

    ex->call_with_validate({t_r}, {t_min_val, t_max_val, t_result_shape, t_use_fixed_seed});
    ASSERT_EQ(t_r->get_shape(), (Shape{500, 2000}));

    auto results_2 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_2.begin(), results_2.end(), [](float x) { return (x <= 10.0f); }));

    ex->call_with_validate({t_r}, {t_min_val, t_max_val, t_result_shape, t_use_fixed_seed});
    auto results_3 = read_vector<float>(t_r);
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x >= -10.0f); }));
    ASSERT_TRUE(
        std::all_of(results_3.begin(), results_3.end(), [](float x) { return (x <= 10.0f); }));

    ASSERT_FALSE(test::all_close_f(results_2, results_3)) << "Two different randomly generated "
                                                             "large vectors matched exactly, even "
                                                             "though use_fixed_seed was not set.";
}
