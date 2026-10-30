// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "pass/simplify_activation_shapes.hpp"

namespace {

using namespace ov::op;

class GGUFActivationShapes : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::frontend::gguf::pass::SimplifyActivationShapes>();
        comparator.enable(FunctionsComparator::ACCURACY);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

std::shared_ptr<ov::Model> residual_model(bool compact) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 8});
    const auto axis = v0::Constant::create(ov::element::i64, {1}, {1});
    const auto w = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(x, w, false, true);
    ov::Output<ov::Node> value = fc;
    ov::Output<ov::Node> residual = x;
    if (!compact) {
        value = std::make_shared<v0::Unsqueeze>(value, axis);
        residual = std::make_shared<v0::Unsqueeze>(residual, axis);
    }
    value = std::make_shared<v4::Swish>(std::make_shared<v1::Add>(value, residual));
    if (compact)
        value = std::make_shared<v0::Unsqueeze>(value, axis);
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, KeepsResidualPathAtRankThree) {
    model = residual_model(false);
    model_ref = residual_model(true);
}

TEST_F(GGUFActivationShapes, PreservesOtherOutputsOfReshapeProducer) {
    const auto make_model = [](bool compact) {
        const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 8});
        const auto split = std::make_shared<v1::VariadicSplit>(x,
                                                               v0::Constant::create(ov::element::i64, {}, {1}),
                                                               v0::Constant::create(ov::element::i64, {2}, {2, 3}));
        ov::Output<ov::Node> second = split->output(1);
        if (compact) {
            second = std::make_shared<v0::Unsqueeze>(second, v0::Constant::create(ov::element::i64, {1}, {1}));
        } else {
            second =
                std::make_shared<v1::Reshape>(second, v0::Constant::create(ov::element::i64, {4}, {1, 1, 3, 8}), false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{split->output(0), second}, ov::ParameterVector{x});
    };
    model = make_model(false);
    model_ref = make_model(true);
}

TEST_F(GGUFActivationShapes, RemovesOnlySymbolProvenIdentityReshapes) {
    const auto make_model = [](bool compact) {
        const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
        ov::Output<ov::Node> value = x;
        if (!compact) {
            const auto expanded = std::make_shared<v0::Unsqueeze>(x, v0::Constant::create(ov::element::i64, {1}, {1}));
            const auto shape = std::make_shared<v3::ShapeOf>(expanded);
            const auto target = std::make_shared<v8::Gather>(shape,
                                                             v0::Constant::create(ov::element::i64, {3}, {0, 2, 3}),
                                                             v0::Constant::create(ov::element::i64, {}, {0}));
            value = std::make_shared<v1::Reshape>(expanded, target, false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v4::Swish>(value)},
                                           ov::ParameterVector{x});
    };
    model = make_model(false);
    model_ref = make_model(true);
}

TEST_F(GGUFActivationShapes, LeavesIndependentDynamicDimensionsInReshape) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
    const auto shape = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{3});
    const auto reshape = std::make_shared<v1::Reshape>(x, shape, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{x, shape});
    comparator.disable(FunctionsComparator::ACCURACY);
}

TEST_F(GGUFActivationShapes, LeavesNonSingletonBroadcastAxisUnchanged) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, -1, 8});
    const auto bias = v0::Constant::create(ov::element::f32, {8}, {0.125f});
    const auto add = std::make_shared<v1::Add>(x, bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, LeavesSelectedTokenAxisDynamic) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 8});
    model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v4::Swish>(x)}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, LeavesUnprovenSingletonAxisWithoutBroadcasting) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 3, 8});
    const auto y = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 1, 3, 8});
    const auto add = std::make_shared<v1::Add>(x, y, ov::op::AutoBroadcastType::NONE);
    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x, y});
}

}  // namespace
