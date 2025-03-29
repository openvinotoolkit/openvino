// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/transformations_after_split.hpp"

#include <string>

#include "openvino/opsets/opset1.hpp"

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> TransformationsAfterSplitFunction::get(const std::string transformationName) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, Shape{1, 9, 16, 16});
    const size_t outputSize = 2ul;

    const auto axis = ov::opset1::Constant::create(ov::element::i64, Shape{}, {2});
    const auto splitLength = ov::opset1::Constant::create(ov::element::i64, Shape{outputSize}, {8, 8});
    const auto variadicSplit = std::make_shared<ov::opset1::VariadicSplit>(input, axis, splitLength);

    ResultVector results;
    for (size_t i = 0; i < outputSize; ++i) {
        const auto additionalLayer = getLayerByTransformationName(transformationName, variadicSplit->output(i));
        results.push_back(std::make_shared<ov::opset1::Result>(additionalLayer));
    }

    const auto function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input },
        "VariadicSplitAndAdditionalLayerTransformation");

    return function;
}

std::shared_ptr<Node> TransformationsAfterSplitFunction::getLayerByTransformationName(
    const std::string transformationName,
    const ov::Output<Node> parent) {
    if (transformationName == "AddTransformationWithoutConcat") {
        const auto dequantization = makeDequantization(parent, { {}, {}, { 3.f } });
        const auto addConstant = ov::opset1::Constant::create(ov::element::u8, Shape{}, {128.f});
        return std::make_shared<ov::opset1::Add>(dequantization, addConstant);
    }
    if (transformationName == "AddTransformationWithConcat") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto addConstant = ov::opset1::Constant::create(ov::element::f32, Shape{}, {128.f});
        return std::make_shared<ov::opset1::Add>(dequantization, addConstant);
    }
    if (transformationName == "AvgPoolTransformation") {
        const auto dequantization = makeDequantization(parent, { {ov::element::f32}, {}, { 0.1f } });
        return std::make_shared<ov::opset1::AvgPool>(dequantization,
                                                     Strides{1, 1},
                                                     Shape{1, 1},
                                                     Shape{0, 0},
                                                     Shape{2, 2},
                                                     true,
                                                     ov::op::RoundingType::FLOOR);
    }
    if (transformationName == "ClampTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return std::make_shared<ov::opset1::Clamp>(dequantization, 0.0, 6.0);
    }
    if (transformationName == "ConvolutionTransformation") {
        const auto dequantizationOnData = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto weights = ov::opset1::Constant::create(ov::element::i8, Shape{3, 9, 1, 1}, {2});
        const auto dequantizationOnWeights = makeDequantization(weights, {{ov::element::f32}, {}, {0.3f}});
        return std::make_shared<ov::opset1::Convolution>(
            dequantizationOnData,
            dequantizationOnWeights,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });
    }
    if (transformationName == "AsymmetricConvolutionTransformation") {
        const auto dequantizationOnData = makeDequantization(parent, {{ov::element::f32}, {128.f}, {0.1f}});
        const auto weights = ov::opset1::Constant::create(ov::element::i8, Shape{3, 9, 1, 1}, {2});
        const auto dequantizationOnWeights = makeDequantization(weights, {{ov::element::f32}, {}, {0.3f}});
        return std::make_shared<ov::opset1::Convolution>(
            dequantizationOnData,
            dequantizationOnWeights,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });
    }
    if (transformationName == "DepthToSpaceTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return std::make_shared<ov::opset1::DepthToSpace>(dequantization, ov::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 3);
    }
    if (transformationName == "FakeQuantizeTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return makeFakeQuantize(dequantization, ov::element::f32, {256, Shape{}, {0.f}, {255.f}, {0.f}, {127.f}});
    }
    if (transformationName == "InterpolateTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto outShape = ov::opset1::Constant::create(ov::element::i64, Shape{4}, {1, 4, 32, 32});

        ov::op::v0::Interpolate::Attributes attributes;
        attributes.axes = ov::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = std::vector<size_t>{ 0ul };
        attributes.pads_end = std::vector<size_t>{ 0ul };

        return std::make_shared<ov::opset1::Interpolate>(dequantization, outShape, attributes);
    }
    if (transformationName == "MatMulTransformation") {
        const auto dequantizationOnData = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto weights = ov::opset1::Constant::create(ov::element::i8, Shape{16, 16}, {2});
        const auto dequantizationOnWeights = makeDequantization(weights, {{ov::element::f32}, {}, {0.3f}});
        return std::make_shared<ov::opset1::MatMul>(dequantizationOnData, dequantizationOnWeights);
    }
    if (transformationName == "MaxPoolTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return std::make_shared<ov::opset1::MaxPool>(
            dequantization,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 });
    }
    if (transformationName == "MultiplyTransformation") {
        const auto dequantization = makeDequantization(parent, {{}, {}, {{2.f}, ov::element::f32, {}}});
        return makeDequantization(dequantization, { {}, {}, { 0.2f } });
    }
    if (transformationName == "MVNTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return std::make_shared<ov::op::v0::MVN>(dequantization, ov::AxisSet{ 2, 3 });
    }
    if (transformationName == "NormalizeL2Transformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto axesNode = ov::opset1::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 3});
        return std::make_shared<ov::opset1::NormalizeL2>(dequantization, axesNode, 1e-6, ov::op::EpsMode::ADD);
    }
    if (transformationName == "PReluTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto slope = std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{}, std::vector<float>{0.1f});
        return std::make_shared<ov::opset1::PRelu>(dequantization, slope);
    }
    if (transformationName == "ReluTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        return std::make_shared<ov::opset1::Relu>(dequantization);
    }
    if (transformationName == "ReshapeTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto reshapeConst = ov::opset1::Constant::create(ov::element::i64, ov::Shape{3}, {1, 3, -1});
        return std::make_shared<ov::opset1::Reshape>(dequantization, reshapeConst, false);
    }
    if (transformationName == "SqueezeTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto squeezeConst = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        return std::make_shared<ov::opset1::Squeeze>(dequantization, squeezeConst);
    }
    if (transformationName == "StridedSliceTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});

        std::vector<int64_t> mask{ 1, 0, 1, 1 };
        const auto beginParam = ov::opset1::Constant::create(ov::element::i64, Shape{4}, {0, 0, 0, 0});
        const auto endParam = ov::opset1::Constant::create(ov::element::i64, Shape{4}, {1, 2, 1, 1});
        const auto stridesParam = ov::opset1::Constant::create(ov::element::i64, Shape{4}, {1, 1, 1, 1});

        return std::make_shared<ov::opset1::StridedSlice>(dequantization, beginParam, endParam, stridesParam, mask, mask);
    }
    if (transformationName == "TransposeTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto transposeConstant = ov::opset1::Constant::create(ov::element::i64, Shape{4}, {0, 1, 3, 2});
        return std::make_shared<ov::opset1::Transpose>(dequantization, transposeConstant);
    }
    if (transformationName == "UnsqueezeTransformation") {
        const auto dequantization = makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
        const auto unsqueezeConst = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        return std::make_shared<ov::opset1::Unsqueeze>(dequantization, unsqueezeConst);
    }
    if (transformationName == "FuseConvertTransformation") {
        return makeDequantization(parent, {{ov::element::f32}, {}, {0.1f}});
    }
    if (transformationName == "FuseSubtractToFakeQuantizeTransformation") {
        // INT8 before FakeQuantize, all operations before FakeQuantize have been fused: need to have TypeRelaxed here
        const auto fakeQuantize =
            makeFakeQuantizeTypeRelaxed(parent, ov::element::f32, {256, Shape{}, {0.f}, {255.f}, {0.f}, {127.f}});
        return makeDequantization(fakeQuantize, {{}, {{128.f}, ov::element::f32, {}}, {}});
    }
    if (transformationName == "FuseMultiplyToFakeQuantizeTransformation") {
        // INT8 before FakeQuantize, all operations before FakeQuantize have been fused: need to have TypeRelaxed here
        const auto fakeQuantize =
            makeFakeQuantizeTypeRelaxed(parent, ov::element::f32, {256, Shape{}, {0.f}, {255.f}, {0.f}, {127.f}});
        return makeDequantization(fakeQuantize, {{}, {}, {{2.f}, ov::element::f32, {}}});
    }
    if (transformationName == "MultiplyToGroupConvolutionTransformation") {
        return makeDequantization(parent, {{}, {{128.f}, ov::element::f32, {}}, {2.f}});
    }
    if (transformationName == "SubtractMultiplyToMultiplyAddTransformation") {
        return makeDequantization(parent, {{}, {{128.f}, ov::element::f32, {}}, {2.f}});
    }
    throw std::runtime_error("unexpected additional layer name");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
