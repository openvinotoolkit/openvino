// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_transition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov;
using namespace testing;

enum class BroadcastVersion { V1, V3 };

std::ostream& operator<<(std::ostream& os, const BroadcastVersion& version) {
    switch (version) {
    case BroadcastVersion::V1:
        return os << "V1";
    case BroadcastVersion::V3:
        return os << "V3";
    default:
        OPENVINO_THROW("Unexpected BroadcastVersion.");
    }
}

std::shared_ptr<ov::Node> getEltwise(
    const ov::Output<ov::Node>& in1,
    const ov::Output<ov::Node>& in2,
    const ov::test::utils::EltwiseTypes& eltwise_type,
    const ov::op::AutoBroadcastType& eltwise_bcast_type = ov::op::AutoBroadcastType::NUMPY) {
    switch (eltwise_type) {
    case ov::test::utils::EltwiseTypes::ADD:
        return std::make_shared<ov::opset10::Add>(in1, in2, eltwise_bcast_type);
    case ov::test::utils::EltwiseTypes::MULTIPLY:
        return std::make_shared<ov::opset10::Multiply>(in1, in2, eltwise_bcast_type);
    case ov::test::utils::EltwiseTypes::SUBTRACT:
        return std::make_shared<ov::opset10::Subtract>(in1, in2, eltwise_bcast_type);
    default:
        OPENVINO_THROW("Unexpected eltwise type");
    }
}

std::shared_ptr<ov::Model> getOriginal(
    const ov::element::Type& precision,
    const ov::PartialShape& input_shape,
    const ov::Shape& target_shape,
    const ov::op::BroadcastType& bcast_mode,
    const BroadcastVersion& bcast_version,
    const ov::test::utils::EltwiseTypes& eltwise_type,
    const size_t idx,
    const ov::op::AutoBroadcastType& eltwise_bcast_type = ov::op::AutoBroadcastType::NUMPY) {
    const auto input = std::make_shared<ov::opset10::Parameter>(precision, input_shape);
    const auto data_constant = ov::opset10::Constant::create(precision, {}, {1.f});
    const auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {target_shape.size()}, target_shape);

    std::shared_ptr<ov::Node> bcast;
    switch (bcast_version) {
    case BroadcastVersion::V1: {
        OPENVINO_ASSERT(bcast_mode != ov::op::BroadcastType::BIDIRECTIONAL,
                        "opset1::Broadcast can't be created with BIDIRECTIONAL mode.");
        static const std::map<ov::op::BroadcastType, ov::op::AutoBroadcastType> bcast_mode_to_autobcast_type{
            {ov::op::BroadcastType::EXPLICIT, ov::op::AutoBroadcastType::EXPLICIT},
            {ov::op::BroadcastType::NONE, ov::op::AutoBroadcastType::NONE},
            {ov::op::BroadcastType::NUMPY, ov::op::AutoBroadcastType::NUMPY},
            {ov::op::BroadcastType::PDPD, ov::op::AutoBroadcastType::PDPD},
        };
        bcast = std::make_shared<ov::op::v1::Broadcast>(data_constant,
                                                        target_shape_node,
                                                        bcast_mode_to_autobcast_type.at(bcast_mode));
        break;
    }
    case BroadcastVersion::V3:
        bcast = std::make_shared<ov::op::v3::Broadcast>(data_constant, target_shape_node, bcast_mode);
        break;
    default:
        OPENVINO_THROW("Unexpected BroadcastVersion.");
    }

    const auto fst_in = idx == 0 ? bcast->output(0) : input->output(0);
    const auto sec_in = idx == 1 ? bcast->output(0) : input->output(0);
    const auto operation = getEltwise(fst_in, sec_in, eltwise_type, eltwise_bcast_type);
    return std::make_shared<ov::Model>(operation, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getReference(const ov::element::Type& precision,
                                        const ov::PartialShape& input_shape,
                                        const ov::Shape& original_target_shape,
                                        const ov::test::utils::EltwiseTypes& eltwise_type,
                                        const size_t idx) {
    const auto input = std::make_shared<ov::opset10::Parameter>(precision, input_shape);
    const auto data_constant = ov::opset10::Constant::create(precision, {}, {1.f});

    const auto fst_in = idx == 0 ? data_constant->output(0) : input->output(0);
    const auto sec_in = idx == 1 ? data_constant->output(0) : input->output(0);
    const auto operation = getEltwise(fst_in, sec_in, eltwise_type, ov::op::AutoBroadcastType::NUMPY);

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

using BroadcastTransitionParams = std::tuple<ov::element::Type,  // precision
                                             ov::Shape,          // input shape
                                             ov::Shape,          // target shape
                                             ov::op::BroadcastType,
                                             BroadcastVersion,
                                             ov::test::utils::EltwiseTypes,
                                             size_t  // broadcast input index
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
        BroadcastVersion bcast_version;
        ov::test::utils::EltwiseTypes eltwise_type;
        size_t idx;
        std::tie(precision, input_shape, target_shape, bcast_mode, bcast_version, eltwise_type, idx) = obj.param;

        std::ostringstream result;
        result << eltwise_type << "_prc=" << precision << "_IS=" << input_shape << "_TS=" << target_shape
               << "_bcast_idx=" << idx << "_bcast_type=" << bcast_mode << "_bcast_version=" << bcast_version;
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        ov::element::Type precision;
        ov::Shape input_shape;
        ov::Shape target_shape;
        ov::op::BroadcastType bcast_mode;
        BroadcastVersion bcast_version;
        ov::test::utils::EltwiseTypes eltwise_type;
        size_t idx;
        std::tie(precision, input_shape, target_shape, bcast_mode, bcast_version, eltwise_type, idx) = GetParam();

        manager.register_pass<ov::pass::BroadcastTransition>();
        model = getOriginal(precision, input_shape, target_shape, bcast_mode, bcast_version, eltwise_type, idx);
        model_ref = getReference(precision, input_shape, target_shape, eltwise_type, idx);
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

std::vector<BroadcastVersion> bcast_versions = {BroadcastVersion::V1, BroadcastVersion::V3};

std::vector<ov::test::utils::EltwiseTypes> operation_types = {ov::test::utils::EltwiseTypes::ADD,
                                                              ov::test::utils::EltwiseTypes::MULTIPLY,
                                                              ov::test::utils::EltwiseTypes::SUBTRACT};
std::vector<size_t> bcast_input_idx = {0, 1};

INSTANTIATE_TEST_SUITE_P(StaticBroadcastTransitionTests_v1_bcast,
                         StaticBroadcastTransitionTests,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes),
                                            ::testing::Values(ov::op::BroadcastType::NUMPY),
                                            ::testing::Values(BroadcastVersion::V1),
                                            ::testing::ValuesIn(operation_types),
                                            ::testing::ValuesIn(bcast_input_idx)),
                         StaticBroadcastTransitionTests::getTestCaseName);

std::vector<ov::op::BroadcastType> all_bcast_modes = {ov::op::BroadcastType::NUMPY,
                                                      ov::op::BroadcastType::BIDIRECTIONAL};

INSTANTIATE_TEST_SUITE_P(StaticBroadcastTransitionTests_v3_bcast,
                         StaticBroadcastTransitionTests,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes),
                                            ::testing::ValuesIn(all_bcast_modes),
                                            ::testing::Values(BroadcastVersion::V3),
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
        const auto operation = getEltwise(input, bcast, ov::test::utils::EltwiseTypes::ADD);
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getEltwise(input, data_constant, ov::test::utils::EltwiseTypes::ADD);
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
        const auto operation = getEltwise(input, bcast, ov::test::utils::EltwiseTypes::ADD);
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getEltwise(input, data_constant, ov::test::utils::EltwiseTypes::ADD);
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
        const auto operation = getEltwise(input, bcast, ov::test::utils::EltwiseTypes::ADD);
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(2));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{4});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getEltwise(input, data_constant, ov::test::utils::EltwiseTypes::ADD);
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
        const auto operation = getEltwise(input, bcast, ov::test::utils::EltwiseTypes::ADD);
        model = std::make_shared<ov::Model>(operation, ov::ParameterVector{input, target_shape});
    }
    manager.register_pass<ov::pass::BroadcastTransition>();
    {
        const auto input = std::make_shared<ov::opset10::Parameter>(data_precision, ov::PartialShape::dynamic(4));
        const auto target_shape = std::make_shared<ov::opset10::Parameter>(shape_precision, ov::PartialShape{2});

        const auto data_constant = ov::opset10::Constant::create(data_precision, {}, {1.f});
        const auto operation = getEltwise(input, data_constant, ov::test::utils::EltwiseTypes::ADD);
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
                        BroadcastVersion::V3,
                        ov::test::utils::EltwiseTypes::ADD,
                        0,
                        ov::op::AutoBroadcastType::EXPLICIT);
    manager.register_pass<ov::pass::BroadcastTransition>();
}

TEST_F(TransformationTestsF, BroadcastTransitionTests_Negative_PDPDEltwiseBcast) {
    model = getOriginal(ov::element::f32,
                        ov::PartialShape{1, 3, 16, 16},
                        ov::Shape{1, 3, 16, 16},
                        ov::op::BroadcastType::NUMPY,
                        BroadcastVersion::V3,
                        ov::test::utils::EltwiseTypes::ADD,
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
