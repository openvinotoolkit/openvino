// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"

TEST(GraphComparatorTests, AllEnablePositiveCheck) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        function = function_ref->clone();
    }
    comparator.enable(FunctionsComparator::NAMES)
        .enable(FunctionsComparator::NODES)
        .enable(FunctionsComparator::CONST_VALUES)
        .enable(FunctionsComparator::PRECISIONS)
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::RUNTIME_KEYS)
        .enable(FunctionsComparator::TENSOR_NAMES);

    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckbyDefault) {
    FunctionsComparator comparator(FunctionsComparator::with_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto add = std::make_shared<ov::op::v1::Add>(input, input2);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input, input2});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {12});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckResultsNumber) {
    FunctionsComparator comparator(FunctionsComparator::with_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto add = std::make_shared<ov::op::v1::Add>(input, input2);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input, input2});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {12});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        auto result1 = std::make_shared<ov::op::v0::Result>(constant);
        auto result2 = std::make_shared<ov::op::v0::Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{input});
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, NamesCheckPositive) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        input->set_friendly_name("new_name1");
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        constant->set_friendly_name("new_name2");
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->set_friendly_name("new_name3");
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        input->set_friendly_name("new_name1");
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        constant->set_friendly_name("new_name2");
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->set_friendly_name("new_name3");
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NAMES).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, NamesCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        input->set_friendly_name("new_name1");
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        constant->set_friendly_name("new_name2");
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->set_friendly_name("new_name3");
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        input->set_friendly_name("new_name1");
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        constant->set_friendly_name("new_name2");
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->set_friendly_name("new_name3_different");
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NAMES).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, ConstCheckWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {12});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, ConstCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {12});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, TensorNamesCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        function = function_ref->clone();
        add->get_input_tensor(0).set_names({"new_name"});
    }
    comparator.enable(FunctionsComparator::TENSOR_NAMES).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, TensorNamesCheckWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        function = function_ref->clone();
        add->get_input_tensor(0).set_names({"new_name"});
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckAttributesNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::op::v0::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::op::v0::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckPrecisionsNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::f32, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::PRECISIONS).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckPrecisionsWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::f32, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfo) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->get_rt_info()["my_info"] = 42;
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfoReverse) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->get_rt_info()["my_info"] = 42;
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfoInput) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->input(0).get_rt_info()["my_info"] = 42;
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfoOutput) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        add->output(0).get_rt_info()["my_info"] = 42;
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {3}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS).enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckTensorIteratorPositive) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 16});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128});

        auto Xi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 16});
        auto Yi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128});

        // Body
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{384, 16}, w_val);
        auto R = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{384, 128}, r_val);
        auto B = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{384}, b_val);

        auto gru_cell = std::make_shared<ov::op::v3::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<ov::op::v0::Result>(gru_cell);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<ov::op::v0::Result>(unsqueeze);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{res_1, res_2}, ov::ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<ov::op::v0::Result>(tensor_iterator->output(1));
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{res_ti_1}, ov::ParameterVector{X, Y});
        function = function_ref->clone();
    }
    comparator.enable(FunctionsComparator::NODES);
    comparator.enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

namespace {
std::shared_ptr<ov::Model> make_check_loop_model(bool different_body) {
    auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Yi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M_body = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 3);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
    // Body
    auto sum = std::make_shared<ov::op::v1::Add>(Xi, Yi);
    std::shared_ptr<ov::Node> Zo;
    if (different_body) {
        auto neg = std::make_shared<ov::op::v0::Negative>(sum);
        Zo = std::make_shared<ov::op::v1::Multiply>(neg, M_body);
    } else {
        Zo = std::make_shared<ov::op::v1::Multiply>(sum, M_body);
    }
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo}, ov::ParameterVector{Xi, Yi, M_body});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    // Output is last Zo
    auto result = std::make_shared<ov::op::v0::Result>(loop->get_iter_value(Zo, -1));
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{X, Y, M});
}
}  // namespace

TEST(GraphComparatorTests, CheckLoopPositive) {
    std::shared_ptr<ov::Model> function, function_ref;
    function_ref = make_check_loop_model(false);
    function = function_ref->clone();

    auto comparator = FunctionsComparator::no_default();
    comparator.enable(FunctionsComparator::NODES);
    comparator.enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);
    const auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckLoopNegative) {
    std::shared_ptr<ov::Model> function, function_ref;
    function_ref = make_check_loop_model(false);
    function = make_check_loop_model(true);

    auto comparator = FunctionsComparator::no_default();
    comparator.enable(FunctionsComparator::NODES);
    comparator.enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);
    const auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid);
}

TEST(GraphComparatorTests, CheckSinksPositive) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto read2 = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto add = std::make_shared<ov::op::v1::Add>(arg, read);
        auto add2 = std::make_shared<ov::op::v1::Add>(arg, read2);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto assign2 = std::make_shared<ov::op::v6::Assign>(add, variable);

        auto res = std::make_shared<ov::op::v0::Result>(add);
        auto res2 = std::make_shared<ov::op::v0::Result>(add2);

        function_ref = std::make_shared<ov::Model>(ov::ResultVector({res, res2}),
                                                   ov::SinkVector({assign, assign2}),
                                                   ov::ParameterVector({arg}));
        function = function_ref->clone();
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckSinksNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto read2 = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto add = std::make_shared<ov::op::v1::Add>(arg, read);
        auto add2 = std::make_shared<ov::op::v1::Add>(arg, read2);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto assign2 = std::make_shared<ov::op::v6::Assign>(add, variable);

        auto res = std::make_shared<ov::op::v0::Result>(add);
        auto res2 = std::make_shared<ov::op::v0::Result>(add2);

        function_ref = std::make_shared<ov::Model>(ov::ResultVector({res, res2}),
                                                   ov::SinkVector({assign, assign2}),
                                                   ov::ParameterVector({arg}));
    }

    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
        const std::string variable_name("variable_different");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto read2 = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        auto add = std::make_shared<ov::op::v1::Add>(arg, read);
        auto add2 = std::make_shared<ov::op::v1::Add>(arg, read2);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto assign2 = std::make_shared<ov::op::v6::Assign>(add, variable);

        auto res = std::make_shared<ov::op::v0::Result>(add);
        auto res2 = std::make_shared<ov::op::v0::Result>(add2);

        function = std::make_shared<ov::Model>(ov::ResultVector({res, res2}),
                                               ov::SinkVector({assign, assign2}),
                                               ov::ParameterVector({arg}));
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, DisableCheck) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        function = function_ref->clone();
    }
    comparator.enable(FunctionsComparator::NODES);
    comparator.disable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckAccuracyPositive) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::ACCURACY);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckAccuracyNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {12});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {200});
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::ACCURACY);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckAccuracyNotEnabled) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::op::v0::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 9, 3, 4, 5, 6, 7, 8, 9, 1, 12, 3, 9, 5, 0, 7, 8, 9, 1, 2, 12, 4, 9, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::op::v0::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NODES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckConsumersCountPositive) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add_1 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto add_2 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto mul = std::make_shared<ov::op::v1::Multiply>(add_1, add_2);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add_1 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto add_2 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto mul = std::make_shared<ov::op::v1::Multiply>(add_1, add_2);
        function = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NODES).enable(FunctionsComparator::CONSUMERS_COUNT);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckConsumersCountNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add_1 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto add_2 = std::make_shared<ov::op::v1::Add>(input, constant);
        auto mul = std::make_shared<ov::op::v1::Multiply>(add_1, add_2);
        function_ref = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto constant_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto constant_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto add_1 = std::make_shared<ov::op::v1::Add>(input, constant_1);
        auto add_2 = std::make_shared<ov::op::v1::Add>(input, constant_2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(add_1, add_2);
        function = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::NODES).enable(FunctionsComparator::CONSUMERS_COUNT);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}
