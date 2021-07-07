// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

using InputShape = PartialShape;
using TargetShape = Shape;

void eliminate_broadcast_test(std::shared_ptr<Function> f, std::shared_ptr<Function> f_ref) {
    pass::Manager manager;
    manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
    manager.run_passes(f);
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class EliminateBroadcastTest: public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const TargetShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input1, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_output_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }
};

class EliminateBroadcastSwapInputsTest: public CommonTestUtils::TestsCommon,
                              public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const TargetShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(broadcast, input1);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_output_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input2, ref_input1);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }
};

class NoEliminateBroadcastTest: public CommonTestUtils::TestsCommon,
                              public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_input_shape, broadcast_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const TargetShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input1, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_input_shape,
                                            const TargetShape & broadcast_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto ref_input_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);
        auto ref_broadcast = std::make_shared<ngraph::opset5::Broadcast>(ref_input2, ref_input_shape_node);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_broadcast);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }
};

class EliminateDynamicBroadcastTest: public CommonTestUtils::TestsCommon,
                              public testing::WithParamInterface<std::tuple<InputShape, InputShape,
                              InputShape, InputShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());
        const auto& broadcast_output_shape = std::get<3>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_output_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const InputShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64,
                                                                             ngraph::Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input1, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2, input_shape_node});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_output_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }
};

class NoEliminateDynamicBroadcastTest: public CommonTestUtils::TestsCommon,
                               public testing::WithParamInterface<std::tuple<InputShape, InputShape, InputShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_input_shape, broadcast_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const InputShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64,
                                                                             ngraph::Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input1, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2, input_shape_node});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_input_shape,
                                            const InputShape & broadcast_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto ref_input_shape_node =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64,
                                                                                 ngraph::Shape{(size_t)(broadcast_shape.rank().get_length())});
        auto ref_broadcast = std::make_shared<ngraph::opset5::Broadcast>(ref_input2, ref_input_shape_node);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_broadcast);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise},
                                                  ngraph::ParameterVector{ref_input1, ref_input2, ref_input_shape_node});
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

INSTANTIATE_TEST_SUITE_P(EliminateBroadcast, EliminateBroadcastTest,
                        testing::Values(std::make_tuple(InputShape{1, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                                        std::make_tuple(InputShape{DYN, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                                        std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{1, 1, 1}, TargetShape{1, 1, 1}),
                                        std::make_tuple(InputShape{1, 2, 3}, InputShape{2, 3}, TargetShape{2, 3}),
                                        std::make_tuple(InputShape{1, 2, 1}, InputShape{1}, TargetShape{1})));

INSTANTIATE_TEST_SUITE_P(EliminateBroadcastSwapInputs, EliminateBroadcastSwapInputsTest,
                        testing::Values(std::make_tuple(InputShape{1, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                                        std::make_tuple(InputShape{DYN, 2, 3}, InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                                        std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{1, 1, 1}, TargetShape{1, 1, 1}),
                                        std::make_tuple(InputShape{1, 2, 3}, InputShape{2, 3}, TargetShape{2, 3}),
                                        std::make_tuple(InputShape{1, 2, 1}, InputShape{1}, TargetShape{1})));

INSTANTIATE_TEST_SUITE_P(NoEliminateBroadcast, NoEliminateBroadcastTest,
                        testing::Values(std::make_tuple(InputShape{1, 2, 1}, InputShape{3}, TargetShape{3}),
                                        std::make_tuple(InputShape{DYN, 2, 3}, InputShape{3, 2, 3}, TargetShape{3, 2, 3}),
                                        std::make_tuple(InputShape{DYN, DYN, DYN}, InputShape{3, 2, 1}, TargetShape{3, 2, 1}),
                                        std::make_tuple(ngraph::PartialShape::dynamic(), InputShape{1, 2, 3}, TargetShape{1, 2, 3}),
                                        std::make_tuple(ngraph::PartialShape::dynamic(), ngraph::PartialShape::dynamic(), TargetShape{1, 2, 3})));

INSTANTIATE_TEST_SUITE_P(EliminateDynamicBroadcast, EliminateDynamicBroadcastTest,
                        testing::Values(std::make_tuple(InputShape{2, 2, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}),
                                        std::make_tuple(InputShape{2, 2, 4}, InputShape{DYN, DYN, DYN}, InputShape{DYN, DYN, DYN}, InputShape{DYN, DYN, DYN}),
                                        std::make_tuple(InputShape{2, 2, 4}, InputShape{2, 2, 4}, InputShape{2, DYN, 4}, InputShape{2, 2, 4})));

INSTANTIATE_TEST_SUITE_P(NoEliminateDynamicBroadcast, NoEliminateDynamicBroadcastTest,
                        testing::Values(std::make_tuple(InputShape{2, 1, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4}),
                                        std::make_tuple(InputShape{2, DYN, 4}, InputShape{2, DYN, 4}, InputShape{2, DYN, 4})));
