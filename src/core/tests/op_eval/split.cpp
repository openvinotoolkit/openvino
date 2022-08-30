// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/split.hpp"

#include <numeric>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, split) {
    const auto data_shape = Shape{3, 8, 3};
    const auto data = make_shared<op::Parameter>(element::i64, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const size_t num_splits = 4;

    auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    auto f = make_shared<Function>(split, ParameterVector{data, axis});

    std::vector<int64_t> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0);

    std::vector<std::vector<int64_t>> expected_results{
        {0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 48, 49, 50, 51, 52, 53},
        {6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59},
        {12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 60, 61, 62, 63, 64, 65},
        {18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 66, 67, 68, 69, 70, 71}};

    HostTensorVector results(num_splits);
    for (auto& result : results) {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::i64>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{1})}));

    for (size_t i = 0; i < num_splits; ++i) {
        EXPECT_EQ(results[i]->get_element_type(), element::i64);
        EXPECT_EQ(results[i]->get_shape(), (Shape{3, 2, 3}));
        EXPECT_EQ(read_vector<int64_t>(results[i]), expected_results[i]);
    }
}

TEST(op_eval, split_neg_axis) {
    const auto data_shape = Shape{2, 1, 4, 1};
    const auto data = make_shared<op::Parameter>(element::i64, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const size_t num_splits = 4;

    auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    auto f = make_shared<Function>(split, ParameterVector{data, axis});

    std::vector<int64_t> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0);

    std::vector<std::vector<int64_t>> expected_results{{0, 4}, {1, 5}, {2, 6}, {3, 7}};

    HostTensorVector results(num_splits);
    for (auto& result : results) {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::i64>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{-2})}));

    for (size_t i = 0; i < num_splits; ++i) {
        EXPECT_EQ(results[i]->get_element_type(), element::i64);
        EXPECT_EQ(results[i]->get_shape(), (Shape{2, 1, 1, 1}));
        EXPECT_EQ(read_vector<int64_t>(results[i]), expected_results[i]);
    }
}

TEST(op_eval, split_boolean_type) {
    const auto data_shape = Shape{2, 1, 2, 1, 2};
    const auto data = make_shared<op::Parameter>(element::boolean, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const size_t num_splits = 2;

    auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    auto f = make_shared<Function>(split, ParameterVector{data, axis});

    std::vector<char> data_vec{true, false, true, false, true, false, true, false};

    std::vector<std::vector<char>> expected_results{{true, false, true, false}, {true, false, true, false}};

    HostTensorVector results(num_splits);
    for (auto& result : results) {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::boolean>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{2})}));

    for (size_t i = 0; i < num_splits; ++i) {
        EXPECT_EQ(results[i]->get_element_type(), element::boolean);
        EXPECT_EQ(results[i]->get_shape(), (Shape{2, 1, 1, 1, 2}));
        EXPECT_EQ(read_vector<char>(results[i]), expected_results[i]);
    }
}

TEST(op_eval, split_1d) {
    const auto data_shape = Shape{8};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const size_t num_splits = 4;

    auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    auto f = make_shared<Function>(split, ParameterVector{data, axis});

    std::vector<float> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0.0f);

    std::vector<std::vector<float>> expected_results{{0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}};

    HostTensorVector results(num_splits);
    for (auto& result : results) {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{0})}));

    for (size_t i = 0; i < num_splits; ++i) {
        EXPECT_EQ(results[i]->get_element_type(), element::f32);
        EXPECT_EQ(results[i]->get_shape(), (Shape{2}));
        EXPECT_EQ(read_vector<float>(results[i]), expected_results[i]);
    }
}
