// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_broadcast3.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

using InputShape = PartialShape;
using TargetShape = Shape;

void convert_broadcast3_test(std::shared_ptr<Model> f, std::shared_ptr<Model> f_ref) {
    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertBroadcast3>();
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class ConvertBroadcast3NUMPYTest : public ov::test::TestsCommon,
                                   public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = opset1::Constant::create(element::i64, Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<opset3::Broadcast>(input, target_shape_node, op::BroadcastType::NUMPY);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    std::shared_ptr<Model> get_reference_broadcast(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = opset1::Constant::create(element::i64, Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<opset1::Broadcast>(input, target_shape_node, op::AutoBroadcastType::NUMPY);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTMulTest : public ov::test::TestsCommon,
                                         public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = opset1::Constant::create(element::i64, Shape{target_shape.size()}, target_shape);
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    std::shared_ptr<Model> get_reference_broadcast(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto const_node = opset1::Constant::create(element::f32, Shape{target_shape}, {1});
        auto mul = std::make_shared<opset1::Multiply>(input, const_node);

        return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, TargetShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());
        const auto& aligned_target_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, aligned_target_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = opset1::Constant::create(element::i64, Shape{target_shape.size()}, target_shape);
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    std::shared_ptr<Model> get_reference_broadcast(const InputShape& input_shape,
                                                   const TargetShape& aligned_target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node =
            opset1::Constant::create(element::i64, Shape{aligned_target_shape.size()}, aligned_target_shape);
        auto broadcast = std::make_shared<opset1::Broadcast>(input, target_shape_node, op::AutoBroadcastType::NUMPY);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastMultiplyTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = std::make_shared<opset1::Parameter>(element::i64, target_shape);
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input, target_shape_node});
    }

    std::shared_ptr<Model> get_reference_broadcast(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto target_shape_node = std::make_shared<opset1::Parameter>(element::i64, target_shape);
        auto constant_one = opset1::Constant::create(element::f32, {1}, {1});
        auto broadcast =
            std::make_shared<opset1::Broadcast>(constant_one, target_shape_node, op::AutoBroadcastType::NUMPY);
        auto mul = std::make_shared<opset1::Multiply>(input, broadcast);
        return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input, target_shape_node});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastLogicalAndTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::boolean, input_shape);
        auto target_shape_node = std::make_shared<opset1::Parameter>(element::i64, target_shape);
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input, target_shape_node});
    }

    std::shared_ptr<Model> get_reference_broadcast(const InputShape& input_shape, const TargetShape& target_shape) {
        auto input = std::make_shared<opset1::Parameter>(element::boolean, input_shape);
        auto target_shape_node = std::make_shared<opset1::Parameter>(element::i64, target_shape);
        auto constant_one = opset1::Constant::create(element::boolean, {1}, {1});
        auto broadcast =
            std::make_shared<opset1::Broadcast>(constant_one, target_shape_node, op::AutoBroadcastType::NUMPY);
        auto mul = std::make_shared<opset1::LogicalAnd>(input, broadcast);
        return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input, target_shape_node});
    }
};

TEST_P(ConvertBroadcast3NUMPYTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTMulTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastMultiplyTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastLogicalAndTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3NUMPY,
                         ConvertBroadcast3NUMPYTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         TargetShape{1, 2, 3, 4, 5}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, TargetShape{8, 3, 64, 64, 64}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, TargetShape{2, 3, 64, 64, 64}),
                                         std::make_tuple(InputShape{3, 1, DYN, 64, 64}, TargetShape{3, 3, 3, 64, 64}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, TargetShape{3, 3, 64, 64, 64}),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN}, TargetShape{3, 3, 64, 64, 3}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, TargetShape{6, 3, 64, 64}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 1, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, TargetShape{8, 3, 64, 64}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, TargetShape{2, 3, 64, 64}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{3, 3, 3, 64}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN}, TargetShape{3, 3, 64, 4}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, TargetShape{5, 3, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, TargetShape{3, 3, 10}),
                                         std::make_tuple(InputShape{2, DYN, 9}, TargetShape{2, 3, 9}),
                                         std::make_tuple(InputShape{3, 3, DYN}, TargetShape{3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT,
                         ConvertBroadcast3BIDIRECTMulTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         TargetShape{1, 2, 3, 4, 5}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, TargetShape{1, 3, 64, 64, 64}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, TargetShape{2, 1, 64, 64, 64}),
                                         std::make_tuple(InputShape{3, 1, DYN, 64, 64}, TargetShape{3, 3, 1, 64, 64}),
                                         std::make_tuple(InputShape{DYN, 1, DYN, 64, DYN}, TargetShape{3, 3, 3, 64, 1}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, TargetShape{3, 3, 64, 1, 64}),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN}, TargetShape{3, 3, 64, 64, 1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 1, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, TargetShape{1, 3, 64, 64}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, TargetShape{2, 1, 64, 64}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{3, 3, 1, 64}),
                                         std::make_tuple(InputShape{DYN, 3, DYN, 64}, TargetShape{3, 3, 64}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN}, TargetShape{3, 3, 64, 1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, TargetShape{5, 3, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, TargetShape{1, 3, 10}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, TargetShape{10}),
                                         std::make_tuple(InputShape{2, DYN, 9}, TargetShape{2, 1, 9}),
                                         std::make_tuple(InputShape{3, 3, DYN}, TargetShape{3, 3, 1})));

INSTANTIATE_TEST_SUITE_P(
    ConvertBroadcast3BIDIRECT,
    ConvertBroadcast3BIDIRECTBroadcastTest,
    testing::Values(
        std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{2, 2, 3, 4, 5}, TargetShape{2, 2, 3, 4, 5}),
        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, TargetShape{3, 3, 64, 64, 64}, TargetShape{3, 3, 64, 64, 64}),
        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, TargetShape{2, 3, 64, 64, 1}, TargetShape{2, 3, 64, 64, 64}),
        std::make_tuple(InputShape{3, 1, DYN, 64, 64}, TargetShape{1, 3, 3, 64, 64}, TargetShape{3, 3, 3, 64, 64}),
        std::make_tuple(InputShape{3, 1, DYN, 64, DYN}, TargetShape{1, 3, 3, 64, 3}, TargetShape{3, 3, 3, 64, 3}),
        std::make_tuple(InputShape{3, 3, 64, DYN, 64}, TargetShape{1, 1, 1, 2, 64}, TargetShape{3, 3, 64, 2, 64}),
        std::make_tuple(InputShape{3, 3, 64, 64, DYN}, TargetShape{3, 3, 64, 64, 3}, TargetShape{3, 3, 64, 64, 3}),
        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 2, 3}, TargetShape{7, 3, 2, 3}),
        std::make_tuple(InputShape{DYN, 3, 64, 64}, TargetShape{3, 3, 64, 64}, TargetShape{3, 3, 64, 64}),
        std::make_tuple(InputShape{2, DYN, 64, 64}, TargetShape{2, 3, 64, 64}, TargetShape{2, 3, 64, 64}),
        std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{1, 3, 1}, TargetShape{3, 3, 3, 64}),
        std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{3, 3, 64}, TargetShape{3, 3, 3, 64}),
        std::make_tuple(InputShape{3, 3, 64, DYN}, TargetShape{64}, TargetShape{3, 3, 64, 64}),
        std::make_tuple(InputShape{DYN, DYN, DYN}, TargetShape{5, 3, 3}, TargetShape{5, 3, 3}),
        std::make_tuple(InputShape{1, 3, DYN}, TargetShape{3, 3, 10}, TargetShape{3, 3, 10}),
        std::make_tuple(InputShape{2, DYN, 9}, TargetShape{2, 2, 1}, TargetShape{2, 2, 9}),
        std::make_tuple(InputShape{3, 3, DYN}, TargetShape{3}, TargetShape{3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT,
                         ConvertBroadcast3BIDIRECTBroadcastMultiplyTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, TargetShape{4}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, TargetShape{3}),
                                         std::make_tuple(InputShape{3, 1, DYN, 64, 64}, TargetShape{2}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, TargetShape{1}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{4}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, TargetShape{3}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, TargetShape{2}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, TargetShape{4}),
                                         std::make_tuple(InputShape{2, DYN, 9}, TargetShape{3}),
                                         std::make_tuple(InputShape{3, 3, DYN}, TargetShape{2})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT,
                         ConvertBroadcast3BIDIRECTBroadcastLogicalAndTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, TargetShape{4}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, TargetShape{3}),
                                         std::make_tuple(InputShape{3, 1, DYN, 64, 64}, TargetShape{2}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, TargetShape{1}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{4}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, TargetShape{3}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, TargetShape{2}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, TargetShape{1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, TargetShape{5}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, TargetShape{4}),
                                         std::make_tuple(InputShape{2, DYN, 9}, TargetShape{3}),
                                         std::make_tuple(InputShape{3, 3, DYN}, TargetShape{2})));

// Broadcast-3 is converted directly to Broadcast-1 for modes NUMPY, NONE and PDPD
TEST(TransformationTests, ConvertBroadcast3WithNumpyModeToBroadcast1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset3::Broadcast>(input1, target_shape, op::BroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto target_shape = std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset1::Broadcast>(input1, target_shape, op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithPDPDModeToBroadcast1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset3::Broadcast>(input1, target_shape, op::BroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto target_shape = std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset1::Broadcast>(input1, target_shape, op::AutoBroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithExplicitModeToBroadcast1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 5, 2});
        auto brodcast_axis = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input1, target_shape, brodcast_axis, op::BroadcastType::EXPLICIT);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 5, 2});
        auto brodcast_axis = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast =
            std::make_shared<opset1::Broadcast>(input1, target_shape, brodcast_axis, op::AutoBroadcastType::EXPLICIT);

        f_ref = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

// Broadcast-3 with mode BIDIRECTIONAL is converted to Broadcast-1,
// when target shape input is Constant and data input has static dimensions at the axis corresponding to "1" in the
// target.
TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToBroadcast1ConstTargetDataF32) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast = std::make_shared<opset3::Broadcast>(input1, target_shape, op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 2});
        auto target_shape = std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset1::Broadcast>(input, target_shape, op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToBroadcast1ConstTargetDataBoolean) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::boolean, Shape{1, 1, 2});
        auto target_shape = opset1::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast = std::make_shared<opset3::Broadcast>(input1, target_shape, op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::boolean, Shape{1, 1, 2});
        auto target_shape = std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<opset1::Broadcast>(input, target_shape, op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

// Broadcast-3 with mode BIDIRECTIONAL is converted to Multiply for not boolean element types,
// when target shape input is Constant and data input has dynamic dimensions at the axis corresponding to "1" in the
// target.
TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToMultiply) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, -1, -1});
        auto const_target_shape =
            std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input1, const_target_shape, op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, -1, -1});
        auto const_target_shape =
            std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        const auto& target_shape = const_target_shape->cast_vector<size_t>();
        auto broadcast =
            std::make_shared<opset1::Multiply>(input, opset1::Constant::create(element::f32, target_shape, {1}));
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}

// Broadcast-3 with mode BIDIRECTIONAL is converted to LogicalAnd for boolean element types,
// when target shape input is Constant and data input has dynamic dimensions at the axis corresponding to "1" in the
// target.
TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToLogicalAnd) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::boolean, PartialShape{1, -1, -1});
        auto const_target_shape =
            std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast =
            std::make_shared<opset3::Broadcast>(input1, const_target_shape, op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertBroadcast3>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::boolean, PartialShape{1, -1, -1});
        auto const_target_shape =
            std::make_shared<opset1::Constant>(element::i64, Shape{3}, std::vector<int64_t>{3, 5, 1});
        const auto& target_shape = const_target_shape->cast_vector<size_t>();
        auto broadcast =
            std::make_shared<opset1::LogicalAnd>(input, opset1::Constant::create(element::boolean, target_shape, {1}));
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<Model>(NodeVector{broadcast}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast")
        << "Transformation ConvertBroadcast3 should keep output names.\n";
}
