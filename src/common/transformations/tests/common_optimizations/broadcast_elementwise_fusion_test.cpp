// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

using InputShape = PartialShape;
using TargetShape = Shape;

void eliminate_broadcast_test(std::shared_ptr<Model> f, std::shared_ptr<Model> f_ref) {
    pass::Manager manager;
    manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    manager.run_passes(f);
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class EliminateBroadcastTest : public ov::test::TestsCommon,
                               public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape,
                                                const InputShape& broadcast_input_shape,
                                                const TargetShape& broadcast_shape) {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto input_shape_node = opset5::Constant::create(element::i64, Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<opset5::Multiply>(input1, broadcast);
        return std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input1, input2});
    }

    std::shared_ptr<Model> get_reference(const InputShape& input_shape, const InputShape& broadcast_output_shape) {
        auto ref_input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto ref_input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<opset5::Multiply>(ref_input1, ref_input2);

        return std::make_shared<ov::Model>(NodeVector{ref_elementwise}, ParameterVector{ref_input1, ref_input2});
    }
};

class EliminateBroadcastSwapInputsTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape,
                                                const InputShape& broadcast_input_shape,
                                                const TargetShape& broadcast_shape) {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto input_shape_node = opset5::Constant::create(element::i64, Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<opset5::Multiply>(broadcast, input1);
        return std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input1, input2});
    }

    std::shared_ptr<Model> get_reference(const InputShape& input_shape, const InputShape& broadcast_output_shape) {
        auto ref_input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto ref_input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<opset5::Multiply>(ref_input2, ref_input1);

        return std::make_shared<ov::Model>(NodeVector{ref_elementwise}, ParameterVector{ref_input1, ref_input2});
    }
};

class NoEliminateBroadcastTest : public ov::test::TestsCommon,
                                 public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_input_shape, broadcast_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape,
                                                const InputShape& broadcast_input_shape,
                                                const TargetShape& broadcast_shape) {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto input_shape_node = opset5::Constant::create(element::i64, Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<opset5::Multiply>(input1, broadcast);
        return std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input1, input2});
    }

    std::shared_ptr<Model> get_reference(const InputShape& input_shape,
                                         const InputShape& broadcast_input_shape,
                                         const TargetShape& broadcast_shape) {
        auto ref_input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto ref_input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto ref_input_shape_node =
            opset5::Constant::create(element::i64, Shape{broadcast_shape.size()}, broadcast_shape);
        auto ref_broadcast = std::make_shared<opset5::Broadcast>(ref_input2, ref_input_shape_node);
        auto ref_elementwise = std::make_shared<opset5::Multiply>(ref_input1, ref_broadcast);

        return std::make_shared<ov::Model>(NodeVector{ref_elementwise}, ParameterVector{ref_input1, ref_input2});
    }
};

class EliminateDynamicBroadcastTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, InputShape, InputShape, InputShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());
        const auto& broadcast_output_shape = std::get<3>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_output_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape,
                                                const InputShape& broadcast_input_shape,
                                                const InputShape& broadcast_shape) {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto input_shape_node =
            std::make_shared<opset5::Parameter>(element::i64, Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto broadcast = std::make_shared<opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<opset5::Multiply>(input1, broadcast);
        return std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input1, input2, input_shape_node});
    }

    std::shared_ptr<Model> get_reference(const InputShape& input_shape, const InputShape& broadcast_output_shape) {
        auto ref_input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto ref_input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<opset5::Multiply>(ref_input1, ref_input2);

        return std::make_shared<ov::Model>(NodeVector{ref_elementwise}, ParameterVector{ref_input1, ref_input2});
    }
};

class NoEliminateDynamicBroadcastTest
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, InputShape, InputShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_input_shape, broadcast_shape);
    }

    std::shared_ptr<Model> get_initial_function(const InputShape& input_shape,
                                                const InputShape& broadcast_input_shape,
                                                const InputShape& broadcast_shape) {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto input_shape_node =
            std::make_shared<opset5::Parameter>(element::i64, Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto broadcast = std::make_shared<opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<opset5::Multiply>(input1, broadcast);
        return std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input1, input2, input_shape_node});
    }

    std::shared_ptr<Model> get_reference(const InputShape& input_shape,
                                         const InputShape& broadcast_input_shape,
                                         const InputShape& broadcast_shape) {
        auto ref_input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto ref_input2 = std::make_shared<opset5::Parameter>(element::f32, broadcast_input_shape);
        auto ref_input_shape_node =
            std::make_shared<opset5::Parameter>(element::i64, Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto ref_broadcast = std::make_shared<opset5::Broadcast>(ref_input2, ref_input_shape_node);
        auto ref_elementwise = std::make_shared<opset5::Multiply>(ref_input1, ref_broadcast);

        return std::make_shared<ov::Model>(NodeVector{ref_elementwise},
                                           ParameterVector{ref_input1, ref_input2, ref_input_shape_node});
    }
};

TEST_P(EliminateBroadcastTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

TEST_P(EliminateBroadcastSwapInputsTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

TEST_P(NoEliminateBroadcastTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

TEST_P(EliminateDynamicBroadcastTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

TEST_P(NoEliminateDynamicBroadcastTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

INSTANTIATE_TEST_SUITE_P(
    EliminateBroadcast,
    EliminateBroadcastTest,
    testing::Values(std::make_tuple(InputShape{1, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                    std::make_tuple(InputShape{DYN, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                    std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{1, 1, 1}, TargetShape{1, 1, 1}),
                    std::make_tuple(InputShape{1, 2, 3}, InputShape{2, 3}, TargetShape{2, 3}),
                    std::make_tuple(InputShape{1, 2, 1}, InputShape{1}, TargetShape{1})));

INSTANTIATE_TEST_SUITE_P(
    EliminateBroadcastSwapInputs,
    EliminateBroadcastSwapInputsTest,
    testing::Values(std::make_tuple(InputShape{1, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                    std::make_tuple(InputShape{DYN, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                    std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{1, 1, 1}, TargetShape{1, 1, 1}),
                    std::make_tuple(InputShape{1, 2, 3}, InputShape{2, 3}, TargetShape{2, 3}),
                    std::make_tuple(InputShape{1, 2, 1}, InputShape{1}, TargetShape{1})));

INSTANTIATE_TEST_SUITE_P(
    NoEliminateBroadcast,
    NoEliminateBroadcastTest,
    testing::Values(std::make_tuple(InputShape{1, 2, 1}, InputShape{3}, TargetShape{3}),
                    std::make_tuple(InputShape{DYN, 2, 3}, InputShape{3, 2, 3}, TargetShape{3, 2, 3}),
                    std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{3, 2, 1}, TargetShape{3, 2, 1}),
                    std::make_tuple(PartialShape::dynamic(), InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                    std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), TargetShape{1, 2, 3})));

INSTANTIATE_TEST_SUITE_P(
    EliminateDynamicBroadcast,
    EliminateDynamicBroadcastTest,
    testing::Values(
        std::make_tuple(InputShape{2, 2, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}),
        std::make_tuple(InputShape{2, 2, 4},
                        InputShape{DYN, DYN, DYN},
                        InputShape{DYN, DYN, DYN},
                        InputShape{DYN, DYN, DYN}),
        std::make_tuple(InputShape{2, 2, 4}, InputShape{2, 2, 4}, InputShape{2, DYN, 4}, InputShape{2, 2, 4})));

INSTANTIATE_TEST_SUITE_P(
    NoEliminateDynamicBroadcast,
    NoEliminateDynamicBroadcastTest,
    testing::Values(std::make_tuple(InputShape{2, 1, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}),
                    std::make_tuple(InputShape{2, DYN, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4})));

TEST_F(TransformationTestsF, BroadcastElementwiseFusionWithShapeOf) {
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto shape_of = std::make_shared<opset5::ShapeOf>(input);
        auto broadcast = std::make_shared<opset5::Broadcast>(input, shape_of);
        auto elementwise = std::make_shared<opset5::Multiply>(input, broadcast);
        model = std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input});

        manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    }

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto elementwise = std::make_shared<opset5::Multiply>(input, input);
        model_ref = std::make_shared<ov::Model>(NodeVector{elementwise}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, BroadcastElementwiseFusionWithShapeOfNeg) {
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto shape_of = std::make_shared<opset5::ShapeOf>(input);
        auto broadcast = std::make_shared<opset5::Broadcast>(input, shape_of);
        auto elementwise = std::make_shared<opset5::Multiply>(input, broadcast);
        model = std::make_shared<ov::Model>(NodeVector{elementwise, broadcast}, ParameterVector{input});

        manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    }
}

TEST_F(TransformationTestsF, BroadcastElementwiseFusionDynShapesDifferentRanks) {
    {
        auto input = std::make_shared<opset5::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
        auto target_shape = std::make_shared<opset5::Parameter>(ov::element::i32, ov::PartialShape{2});
        auto constant = opset5::Constant::create(ov::element::f32, {}, {1.f});
        auto broadcast = std::make_shared<opset5::Broadcast>(constant, target_shape);
        auto elementwise = std::make_shared<opset5::Add>(input, broadcast);
        model = std::make_shared<ov::Model>(ov::NodeVector{elementwise}, ov::ParameterVector{input, target_shape});

        manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    }
}
