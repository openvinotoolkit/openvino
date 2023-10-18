// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_transition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
using namespace ov;
using namespace testing;

std::shared_ptr<ov::Node> getOperation(
    const ov::Output<ov::Node>& in1,
    const ov::Output<ov::Node>& in2,
    const std::string& operation_type,
    const ov::op::AutoBroadcastType& eltwise_bcast_type = ov::op::AutoBroadcastType::NUMPY) {
    if (operation_type == "Add") {
        return std::make_shared<ov::opset10::Add>(in1, in2, eltwise_bcast_type);
    } else if (operation_type == "Multiply") {
        return std::make_shared<ov::opset10::Multiply>(in1, in2, eltwise_bcast_type);
    } else if (operation_type == "Subtract") {
        return std::make_shared<ov::opset10::Subtract>(in1, in2, eltwise_bcast_type);
    } else {
        throw std::runtime_error("Unexpected operation type");
    }
}

std::shared_ptr<ov::Model> getOriginal(
    const ov::element::Type& precision,
    const ov::PartialShape& input_shape,
    const ov::Shape& target_shape,
    const ov::op::BroadcastType& bcast_mode,
    const std::string& operation_type,
    const size_t idx,
    const ov::op::AutoBroadcastType& eltwise_bcast_type = ov::op::AutoBroadcastType::NUMPY) {
    const auto input = std::make_shared<ov::opset10::Parameter>(precision, input_shape);
    const auto data_constant = ov::opset10::Constant::create(precision, {}, {1.f});
    const auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {target_shape.size()}, target_shape);
    const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape_node, bcast_mode);

    const auto fst_in = idx == 0 ? bcast->output(0) : input->output(0);
    const auto sec_in = idx == 1 ? bcast->output(0) : input->output(0);
    const auto operation = getOperation(fst_in, sec_in, operation_type, eltwise_bcast_type);
    return std::make_shared<ov::Model>(operation, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getReference(const ov::element::Type& precision,
                                        const ov::PartialShape& input_shape,
                                        const ov::Shape& original_target_shape,
                                        const std::string& operation_type,
                                        const size_t idx) {
    const auto input = std::make_shared<ov::opset10::Parameter>(precision, input_shape);
    const auto data_constant = ov::opset10::Constant::create(precision, {}, {1.f});

    const auto fst_in = idx == 0 ? data_constant->output(0) : input->output(0);
    const auto sec_in = idx == 1 ? data_constant->output(0) : input->output(0);
    const auto operation = getOperation(fst_in, sec_in, operation_type, ov::op::AutoBroadcastType::NUMPY);

    const auto target_shape = [&]() {
        auto new_shape = original_target_shape;
        auto op_shape = operation->get_shape();
        while (new_shape.size() < op_shape.size())
            new_shape.insert(new_shape.begin(), 1);
        while (op_shape.size() < new_shape.size())
            op_shape.insert(op_shape.begin(), 1);

        for (size_t i = 0; i < new_shape.size(); ++i) {
            new_shape[i] = std::max(new_shape[i], op_shape[i]);
        }
        return new_shape;
    }();

    const auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {target_shape.size()}, target_shape);
    const auto bcast = std::make_shared<ov::opset10::Broadcast>(operation, target_shape_node);
    return std::make_shared<ov::Model>(bcast, ov::ParameterVector{input});
}

using BroadcastTransitionParams = std::tuple<ov::element::Type,      // precision
                                             ov::Shape,              // input shape
                                             ov::Shape,              // target shape
                                             ov::op::BroadcastType,  // broadcast mode
                                             std::string,            // operation type
                                             size_t                  // broadcast input index
                                             >;

class StaticBroadcastTransitionTests : public testing::WithParamInterface<BroadcastTransitionParams>,
                                       public TransformationTestsF {
public:
    StaticBroadcastTransitionTests() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastTransitionParams>& obj) {
        ov::element::Type precision;
        ov::Shape input_shape;
        ov::Shape target_shape;
        ov::op::BroadcastType bcast_mode;
        std::string operation_type;
        size_t idx;
        std::tie(precision, input_shape, target_shape, bcast_mode, operation_type, idx) = obj.param;

        std::ostringstream result;
        result << operation_type << "_prc=" << precision << "_IS=" << input_shape << "_TS=" << target_shape
               << "_bcast_idx=" << idx << "_bcast_type=" << bcast_mode;
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        ov::element::Type precision;
        ov::Shape input_shape;
        ov::Shape target_shape;
        ov::op::BroadcastType bcast_mode;
        std::string operation_type;
        size_t idx;
        std::tie(precision, input_shape, target_shape, bcast_mode, operation_type, idx) = GetParam();

        manager.register_pass<ov::pass::BroadcastTransition>();
        model = getOriginal(precision, input_shape, target_shape, bcast_mode, operation_type, idx);
        model_ref = getReference(precision, input_shape, target_shape, operation_type, idx);
    }
};

TEST_P(StaticBroadcastTransitionTests, BroadcastTransition) {}

namespace BroadcastTransitionTestsInstantiation {
std::vector<ov::Shape> input_shapes = {
    {1, 3, 16, 16},
    {1, 3, 1, 16},
    {16, 16},
};

std::vector<ov::Shape> target_shapes = {
    {1, 3, 16, 1},
    {16, 16},
};

std::vector<ov::op::BroadcastType> bcast_modes = {ov::op::BroadcastType::NUMPY, ov::op::BroadcastType::BIDIRECTIONAL};

std::vector<std::string> operation_types = {"Add", "Multiply", "Subtract"};
std::vector<size_t> bcast_input_idx = {0, 1};

INSTANTIATE_TEST_SUITE_P(TransformationTestsF,
                         StaticBroadcastTransitionTests,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes),
                                            ::testing::ValuesIn(bcast_modes),
                                            ::testing::ValuesIn(operation_types),
                                            ::testing::ValuesIn(bcast_input_idx)),
                         StaticBroadcastTransitionTests::getTestCaseName);
}  // namespace BroadcastTransitionTestsInstantiation

TEST_F(TransformationTestsF, BroadcastTransitionTests_Dynamic_U32TargetShapePrecision) {
    const auto data_precision = ov::element::f32;
    const auto shape_precision = ov::element::u32;
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape);
        const auto operation = getOperation(input, bcast, "Add");
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getOperation(input, data_constant, "Add");
        const auto shapeof = std::make_shared<ov::opset10::ShapeOf>(operation);
        const auto convert = std::make_shared<ov::opset10::Convert>(shapeof, shape_precision);
        const auto max = std::make_shared<ov::opset10::Maximum>(convert, target_shape);
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(operation, max);
        model_ref = std::make_shared<ov::Model>(bcast, ov::ParameterVector{input, target_shape});
    }
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Dynamic_EqualRanks) {
    const auto data_precision = ov::element::f32;
    const auto shape_precision = ov::element::i32;
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape);
        const auto operation = getOperation(input, bcast, "Add");
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getOperation(input, data_constant, "Add");
        const auto shapeof = std::make_shared<ov::opset10::ShapeOf>(operation, shape_precision);
        const auto max = std::make_shared<ov::opset10::Maximum>(shapeof, target_shape);
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(operation, max);
        model_ref = std::make_shared<ov::Model>(bcast, ov::ParameterVector{input, target_shape});
    }
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Dynamic_DataRankLessThanTarget) {
    const auto data_precision = ov::element::f32;
    const auto shape_precision = ov::element::i32;
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(2));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape);
        const auto operation = getOperation(input, bcast, "Add");
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(2));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getOperation(input, data_constant, "Add");
        const auto shapeof = std::make_shared<ov::opset10::ShapeOf>(operation, shape_precision);
        const auto constant = ov::opset10::Constant::create(shape_precision, {2}, {1});
        const auto concat = std::make_shared<ov::opset10::Concat>(ov::OutputVector{constant, shapeof}, 0);
        const auto max = std::make_shared<ov::opset10::Maximum>(concat, target_shape);
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(operation, max);
        model_ref = std::make_shared<ov::Model>(bcast, ov::ParameterVector{input, target_shape});
    }
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Dynamic_DataRankGreaterThanTarget) {
    const auto data_precision = ov::element::f32;
    const auto shape_precision = ov::element::i32;
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{2});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape);
        const auto operation = getOperation(input, bcast, "Add");
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{2});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getOperation(input, data_constant, "Add");
        const auto shapeof = std::make_shared<ov::opset10::ShapeOf>(operation, shape_precision);
        const auto constant = ov::opset10::Constant::create(shape_precision, {2}, {1});
        const auto concat = std::make_shared<ov::opset10::Concat>(ov::OutputVector{constant, target_shape}, 0);
        const auto max = std::make_shared<ov::opset10::Maximum>(shapeof, concat);
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(operation, max);
        model_ref = std::make_shared<ov::Model>(bcast, ov::ParameterVector{input, target_shape});
    }
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_ExplicitEltwiseBcast) {
    model = getOriginal(ov::element::f32,
                        ov::PartialShape{1, 3, 16, 16},
                        ov::Shape{1, 3, 16, 16},
                        ov::op::BroadcastType::NUMPY,
                        "Add",
                        0,
                        ov::op::AutoBroadcastType::EXPLICIT);
    manager.register_pass<ov::pass::BroadcastTransition>();
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_PDPDEltwiseBcast) {
    model = getOriginal(ov::element::f32,
                        ov::PartialShape{1, 3, 16, 16},
                        ov::Shape{1, 3, 16, 16},
                        ov::op::BroadcastType::NUMPY,
                        "Add",
                        0,
                        ov::op::AutoBroadcastType::PDPD);
    manager.register_pass<ov::pass::BroadcastTransition>();
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_PDPDBcastType) {
    const auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});

    const auto data_constant = ov::opset10::Constant::create(ov::element::f32, {1, 1, 1}, {1.f});
    const auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {3}, {1, 16, 16});
    const ov::op::BroadcastModeSpec pdpd_spec(ov::op::BroadcastType::PDPD);
    const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape_node, pdpd_spec);
    const auto add = std::make_shared<ov::opset10::Add>(input, bcast);

    model = std::make_shared<ov::Model>(add, ov::ParameterVector{input});
    manager.register_pass<ov::pass::BroadcastTransition>();
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_WithAxesMapping) {
    const auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});
    const auto data_constant = ov::opset10::Constant::create(ov::element::f32, {16, 16}, {1.f});

    const auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {3}, {1, 16, 16});
    const auto axes_node = ov::opset10::Constant::create(ov::element::i32, {2}, {1, 2});
    const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape_node, axes_node);
    const auto add = std::make_shared<ov::opset10::Add>(input, bcast);

    model = std::make_shared<ov::Model>(add, ov::ParameterVector{input});
    manager.register_pass<ov::pass::BroadcastTransition>();
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_DynamicRank) {
    const auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto data_constant = ov::opset10::Constant::create(ov::element::f32, {}, {1.f});

    const auto target_shape_input = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::PartialShape{-1});
    const auto bcast = std::make_shared<ov::opset10::Broadcast>(data_constant, target_shape_input);
    const auto add = std::make_shared<ov::opset10::Add>(input, bcast);

    model = std::make_shared<ov::Model>(add, ov::ParameterVector{input, target_shape_input});
    manager.register_pass<ov::pass::BroadcastTransition>();
}
