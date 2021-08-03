// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T>
bool compare_set(const vector<T>& a, vector<T> b)
{
    for (auto ita = a.begin(); ita != a.end(); ++ita)
    {
        auto itb = find(b.begin(), b.end(), *ita);
        if (itb == b.end())
        {
            return false;
        }
        else
        {
            b.erase(itb);
        }
    }
    return true;
}

NGRAPH_TEST(${BACKEND_NAME}, topk_resnet50)
{
    Shape shape{128, 1000};
    Shape rshape5{128, 5};
    Shape rshape1{128, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::v1::TopK>(A,
                                       op::Constant::create(element::i64, {}, {5}),
                                       1,
                                       op::v1::TopK::Mode::MAX,
                                       op::v1::TopK::SortType::SORT_VALUES);
    auto C = make_shared<op::v1::TopK>(A,
                                       op::Constant::create(element::i64, {}, {1}),
                                       1,
                                       op::v1::TopK::Mode::MAX,
                                       op::v1::TopK::SortType::SORT_VALUES);

    auto out5_value = B->output(0);
    auto out5_index = B->output(1);
    auto out1_value = C->output(0);
    auto out1_index = C->output(1);
    auto f = make_shared<Function>(OutputVector{out5_value, out5_index, out1_value, out1_index},
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result5_value = backend->create_tensor(element::f32, rshape5);
    auto result5_index = backend->create_tensor(element::i32, rshape5);
    auto result1_value = backend->create_tensor(element::f32, rshape1);
    auto result1_index = backend->create_tensor(element::i32, rshape1);

    auto exec = backend->compile(f);
    exec->call({result5_value, result5_index, result1_value, result1_index}, {a});

    auto actual5_value = read_vector<float>(result5_value);
    auto actual5_index = read_vector<int32_t>(result5_index);
    auto actual1_value = read_vector<float>(result1_value);
    auto actual1_index = read_vector<int32_t>(result1_index);

    vector<float> expected5_value;
    vector<int32_t> expected5_index;
    for (size_t i = 0; i < rshape5[0]; i++)
    {
        for (size_t j = 0; j < rshape5[1]; j++)
        {
            expected5_value.push_back(shape[1] - j - 1);
            expected5_index.push_back(shape[1] - j - 1);
        }
    }

    vector<float> expected1_value;
    vector<int32_t> expected1_index;
    for (size_t i = 0; i < rshape1[0]; i++)
    {
        for (size_t j = 0; j < rshape1[1]; j++)
        {
            expected1_value.push_back(shape[1] - j - 1);
            expected1_index.push_back(shape[1] - j - 1);
        }
    }

    EXPECT_TRUE(compare_set<float>(expected5_value, actual5_value));
    EXPECT_TRUE(compare_set<int32_t>(expected5_index, actual5_index));
    EXPECT_TRUE(compare_set<float>(expected1_value, actual1_value));
    EXPECT_TRUE(compare_set<int32_t>(expected1_index, actual1_index));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_max_sort_none)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::NONE);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    for (size_t i = 0; i < rshape[0]; i++)
    {
        vector<float> expected_value;
        vector<int32_t> expected_index;
        vector<float> act_value;
        vector<int32_t> act_index;
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(shape[1] - j - 1);
            expected_index.push_back(shape[1] - j - 1);
            act_value.push_back(actual_value[rshape[1] * i + j]);
            act_index.push_back(actual_index[rshape[1] * i + j]);
        }
        EXPECT_TRUE(compare_set<float>(expected_value, act_value));
        EXPECT_TRUE(compare_set<int32_t>(expected_index, act_index));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_min_sort_none)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::NONE);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    for (size_t i = 0; i < rshape[0]; i++)
    {
        vector<float> expected_value;
        vector<int32_t> expected_index;
        vector<float> act_value;
        vector<int32_t> act_index;
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(j);
            expected_index.push_back(j);
            act_value.push_back(actual_value[rshape[1] * i + j]);
            act_index.push_back(actual_index[rshape[1] * i + j]);
        }
        EXPECT_TRUE(compare_set<float>(expected_value, act_value));
        EXPECT_TRUE(compare_set<int32_t>(expected_index, act_index));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_max_sort_value)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    vector<float> expected_value;
    vector<int32_t> expected_index;
    for (size_t i = 0; i < rshape[0]; i++)
    {
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(shape[1] - j - 1);
            expected_index.push_back(shape[1] - j - 1);
        }
    }
    EXPECT_TRUE(test::all_close_f(expected_value, actual_value));
    EXPECT_EQ(expected_index, actual_index);
}

NGRAPH_TEST(${BACKEND_NAME}, topk_min_sort_value)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    for (size_t i = 0; i < rshape[0]; i++)
    {
        vector<float> expected_value;
        vector<int32_t> expected_index;
        vector<float> act_value;
        vector<int32_t> act_index;
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(j);
            expected_index.push_back(j);
            act_value.push_back(actual_value[rshape[1] * i + j]);
            act_index.push_back(actual_index[rshape[1] * i + j]);
        }
        EXPECT_TRUE(compare_set<float>(expected_value, act_value));
        EXPECT_TRUE(compare_set<int32_t>(expected_index, act_index));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_max_sort_index)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_INDICES);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    for (size_t i = 0; i < rshape[0]; i++)
    {
        vector<float> expected_value;
        vector<int32_t> expected_index;
        vector<float> act_value;
        vector<int32_t> act_index;
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(shape[1] - j - 1);
            expected_index.push_back(shape[1] - j - 1);
            act_value.push_back(actual_value[rshape[1] * i + j]);
            act_index.push_back(actual_index[rshape[1] * i + j]);
        }
        EXPECT_TRUE(compare_set<float>(expected_value, act_value));
        EXPECT_TRUE(compare_set<int32_t>(expected_index, act_index));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_min_sort_index)
{
    Shape shape{128, 1000};
    Shape rshape{128, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {5});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_INDICES);
    auto out_value = B->output(0);
    auto out_index = B->output(1);
    auto f = make_shared<Function>(OutputVector{out_value, out_index}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> data;
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            data.push_back(j);
        }
    }
    copy_data(a, data);

    auto result_value = backend->create_tensor(element::f32, rshape);
    auto result_index = backend->create_tensor(element::i32, rshape);

    auto exec = backend->compile(f);
    exec->call({result_value, result_index}, {a});

    auto actual_value = read_vector<float>(result_value);
    auto actual_index = read_vector<int32_t>(result_index);

    for (size_t i = 0; i < rshape[0]; i++)
    {
        vector<float> expected_value;
        vector<int32_t> expected_index;
        vector<float> act_value;
        vector<int32_t> act_index;
        for (size_t j = 0; j < rshape[1]; j++)
        {
            expected_value.push_back(j);
            expected_index.push_back(j);
            act_value.push_back(actual_value[rshape[1] * i + j]);
            act_index.push_back(actual_index[rshape[1] * i + j]);
        }
        EXPECT_TRUE(compare_set<float>(expected_value, act_value));
        EXPECT_TRUE(compare_set<int32_t>(expected_index, act_index));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {6});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6, 5, 4, 3, 2, 1}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_i32_max_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto k = op::Constant::create(element::i64, {}, {6});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_EQ((vector<int32_t>{6, 5, 4, 3, 2, 1}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {3});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6, 5, 4}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {6});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {3});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {3});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10, 12, 9, 4, 8, 2, 11, 7, 6, 3, 5, 1}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_int64)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {3});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, element::i64);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i64, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10, 12, 9, 4, 8, 2, 11, 7, 6, 3, 5, 1}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int64_t>{1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0}), read_vector<int64_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_5d_max_partial)
{
    Shape shape{2, 6, 3, 2, 4};
    Shape rshape{2, 2, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a,
        vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 27.,  150., 222., 158., 230., 166., 23.,
            151., 223., 159., 231., 17.,  39.,  2.,   224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 25.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 24.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  10.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  20.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 27.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.});

    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{169, 241, 177, 249, 185, 233, 170, 242, 178, 250, 186, 258, 171, 243,
                       179, 251, 187, 259, 172, 224, 180, 252, 188, 260, 149, 221, 157, 229,
                       165, 113, 150, 222, 158, 230, 166, 234, 151, 223, 159, 231, 163, 235,
                       148, 220, 160, 232, 168, 240, 197, 269, 205, 277, 213, 285, 198, 270,
                       206, 278, 214, 286, 199, 271, 207, 279, 215, 287, 200, 272, 241, 280,
                       216, 288, 193, 265, 201, 273, 209, 281, 194, 266, 202, 274, 210, 262,
                       175, 127, 183, 255, 191, 263, 176, 248, 208, 256, 212, 284}),
        read_vector<float>(result0),
        MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ(
        (vector<int32_t>{5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5,
                         3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3,
                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5,
                         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5, 1, 3, 3}),
        read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10, 12, 9, 4, 11, 7, 6, 3}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 2, 0, 1}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{10, 12, 11, 7}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 2, 2}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {3});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{8, 2, 10, 4, 12, 9, 5, 1, 6, 3, 11, 7}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 0, 1, 1, 0, 0, 1, 2, 2}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{8, 2, 10, 4, 5, 1, 6, 3}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 1, 0, 0, 1}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{8, 2, 5, 1}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {4});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{12, 11, 10, 9, 8, 7, 6, 2, 5, 3, 1, 4}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3, 2, 0, 2, 3, 2, 1}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{12, 11, 10, 9, 8, 7}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{12, 11, 10}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_one_with_equal_values)
{
    Shape shape{2, 4};
    Shape rshape{2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 3, 2, 4, 1, 3, 3, 2});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{4, 3}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{3, 1}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {4});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 1, 4, 6, 2, 5, 9, 8, 7, 12, 11, 10}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2, 1, 1, 3, 0, 3, 0}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{3, 1, 4, 6, 2, 5}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {1});
    int64_t axis = 0;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::NONE);
    auto f0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto f1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto result1 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{3, 1, 4}), read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
    auto h1 = backend->compile(f1);
    h1->call_with_validate({result1}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1}), read_vector<int32_t>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_large_input_max)
{
    Shape shape{4, 8192, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto k = op::Constant::create(element::i64, {}, {10});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);

    auto interp_f_0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto interp_f_1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});
    auto gpu_f_0 = ngraph::clone_function(*interp_f_0);
    auto gpu_f_1 = ngraph::clone_function(*interp_f_1);

    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : interp_f_0->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        iota(tensor_val.begin(), tensor_val.end(), 0.0f);
        args.push_back(tensor_val);
    }

    auto interp_results_0 = execute<float>(interp_f_0, args, "INTERPRETER");
    auto gpu_results_0 = execute<float>(gpu_f_0, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < gpu_results_0.size(); i++)
    {
        EXPECT_TRUE(test::all_close_f(
            gpu_results_0.at(i), interp_results_0.at(i), MIN_FLOAT_TOLERANCE_BITS));
    }

    auto interp_results_1 = execute<float, int32_t>(interp_f_1, args, "INTERPRETER");
    auto gpu_results_1 = execute<float, int32_t>(gpu_f_1, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < gpu_results_1.size(); i++)
    {
        EXPECT_EQ(gpu_results_1.at(i), interp_results_1.at(i));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_large_input_min)
{
    Shape shape{4, 8192, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto k = op::Constant::create(element::i64, {}, {10});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);

    auto interp_f_0 = make_shared<Function>(OutputVector{B->output(0)}, ParameterVector{A});
    auto interp_f_1 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});
    auto gpu_f_0 = ngraph::clone_function(*interp_f_0);
    auto gpu_f_1 = ngraph::clone_function(*interp_f_1);

    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : interp_f_0->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        iota(tensor_val.begin(), tensor_val.end(), 0.0f);
        args.push_back(tensor_val);
    }

    auto interp_results_0 = execute(interp_f_0, args, "INTERPRETER");
    auto gpu_results_0 = execute(gpu_f_0, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < gpu_results_0.size(); i++)
    {
        EXPECT_TRUE(test::all_close_f(
            gpu_results_0.at(i), interp_results_0.at(i), MIN_FLOAT_TOLERANCE_BITS));
    }

    auto interp_results_1 = execute<float, int32_t>(interp_f_1, args, "INTERPRETER");
    auto gpu_results_1 = execute<float, int32_t>(gpu_f_1, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < gpu_results_1.size(); i++)
    {
        EXPECT_EQ(gpu_results_1.at(i), interp_results_1.at(i));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_single_output)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = op::Constant::create(element::i64, {}, {2});
    int64_t axis = 1;
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MIN, op::v1::TopK::SortType::SORT_VALUES);
    auto f0 = make_shared<Function>(OutputVector{B->output(1)}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    h0->call_with_validate({result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 1, 0, 0, 1}), read_vector<int32_t>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_v1_invalid_strings)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto k = op::Constant::create(element::i64, Shape{}, {1});
    EXPECT_THROW(op::v1::TopK(data, k, 0, "max", "invalid_mode"), ngraph::CheckFailure);
    EXPECT_THROW(op::v1::TopK(data, k, 0, "invalid_sort", "index"), ngraph::CheckFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, topk_v1_invalid_k)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

    // K must be a scalar
    const auto k_non_scalar = op::Constant::create(element::i64, Shape{2}, {1, 2});
    EXPECT_THROW(op::v1::TopK(data, k_non_scalar, 0, "max", "index"),
                 ngraph::NodeValidationFailure);

    // K can only be i8, i32 or i64
    const auto k_float = op::Constant::create(element::f32, Shape{}, {1.0f});
    EXPECT_THROW(op::v1::TopK(data, k_float, 0, "max", "index"), ngraph::NodeValidationFailure);

    // the value of K must be positive
    const auto k_negative = op::Constant::create(element::i8, Shape{}, {-1});
    EXPECT_THROW(op::v1::TopK(data, k_negative, 0, "max", "index"), ngraph::NodeValidationFailure);
}

template <typename T>
class topk_backend : public ::testing::Test
{
};
TYPED_TEST_SUITE_P(topk_backend);

template <typename Mode, typename SortType>
struct TopkSortTestOutputs
{
    Mode mode;
    SortType sort_type;
    std::vector<float> output0;
    std::vector<int32_t> output1;

    TopkSortTestOutputs(Mode mode,
                        SortType sort_type,
                        std::vector<float>&& output0,
                        std::vector<int32_t>&& output1)
        : mode(mode)
        , sort_type(sort_type)
        , output0(std::move(output0))
        , output1(std::move(output1))
    {
    }
};

TYPED_TEST_P(topk_backend, topk_mode_sort_order)
{
    const Shape shape{5};
    const Shape rshape{3};
    const auto data = make_shared<op::Parameter>(element::f32, shape);
    const auto k = op::Constant::create(element::i64, {}, {3});
    const int64_t axis = 0;

    // helpers to reduce code verbosity
    using m = typename TypeParam::Mode;
    using st = typename TypeParam::SortType;
    using v_f = std::vector<float>;
    using v_i = std::vector<int32_t>;

    const std::vector<float> input{3, 1, 2, 5, 4};

    std::vector<TopkSortTestOutputs<m, st>> valid_outputs;
    valid_outputs.emplace_back(m::MAX, st::SORT_VALUES, v_f{5, 4, 3}, v_i{3, 4, 0});
    valid_outputs.emplace_back(m::MAX, st::SORT_INDICES, v_f{3, 5, 4}, v_i{0, 3, 4});
    valid_outputs.emplace_back(m::MIN, st::SORT_VALUES, v_f{1, 2, 3}, v_i{1, 2, 0});
    valid_outputs.emplace_back(m::MIN, st::SORT_INDICES, v_f{3, 1, 2}, v_i{0, 1, 2});

    for (const auto& v : valid_outputs)
    {
        auto topk = make_shared<TypeParam>(data, k, axis, v.mode, v.sort_type);
        auto f0 = make_shared<Function>(OutputVector{topk->output(0)}, ParameterVector{data});
        auto f1 = make_shared<Function>(OutputVector{topk->output(1)}, ParameterVector{data});

        auto test_case0 = test::TestCase<TestEngine>(f0);
        test_case0.add_input<float>(input);
        test_case0.add_expected_output<float>(rshape, v.output0);
        test_case0.run();

        auto test_case1 = test::TestCase<TestEngine>(f1);
        test_case1.add_input<float>(input);
        test_case1.add_expected_output<int32_t>(rshape, v.output1);
        test_case1.run();
    }
}

REGISTER_TYPED_TEST_SUITE_P(topk_backend, topk_mode_sort_order);

typedef ::testing::Types<op::v1::TopK, op::v3::TopK> TopKTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(${BACKEND_NAME}, topk_backend, TopKTypes);
