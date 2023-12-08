// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/builders.hpp"

#include <queue>
#include <memory>

#include <openvino/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ov::pass::low_precision;

std::shared_ptr<Node> makeDequantization(
    const Output<Node>& data,
    const DequantizationOperations& dequantizationOperations) {
    Output<Node> parent = data;

    if (!dequantizationOperations.convert.empty()) {
        auto convert = std::make_shared<ov::opset1::Convert>(data, dequantizationOperations.convert.outPrecision);
        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), convert }, convert);
        convert->set_friendly_name(data.get_node_shared_ptr()->get_friendly_name() + "/DequantizationConvert");
        parent = convert;
    }

    if (!dequantizationOperations.subtract.empty()) {
        std::shared_ptr<ov::opset1::Subtract> subtract;

        std::vector<size_t> shape;
        auto values = dequantizationOperations.subtract.values;
        if (dequantizationOperations.subtract.constantShapeIsDefined) {
            shape = dequantizationOperations.subtract.constantShape;
            if (values.size() == 1ul) {
                values = std::vector<float>(shape_size(shape), values[0]);
            }
        } else {
            if (dequantizationOperations.subtract.values.size() == 1ul) {
                shape = std::vector<size_t>({});
            } else {
                const auto rank = parent.get_partial_shape().rank();
                shape = std::vector<size_t>(rank.is_dynamic() ? 4ul : rank.get_length(), 1ul);
                shape[shape.size() >= 2 ? 1ul : 0] = dequantizationOperations.subtract.values.size();
            }
        }

        std::shared_ptr<Node> subtractConst = std::make_shared<ov::opset1::Constant>(
            dequantizationOperations.subtract.constantPrecision != element::undefined ?
                dequantizationOperations.subtract.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        if (dequantizationOperations.subtract.addConvert) {
            std::shared_ptr<Node> subtractConstConvert = std::make_shared<ov::opset1::Convert>(
                subtractConst,
                dequantizationOperations.subtract.outPrecision == element::undefined ?
                    parent.get_element_type() :
                    dequantizationOperations.subtract.outPrecision);

            auto& rt = subtractConstConvert->get_rt_info();
            for (const auto& attribute : dequantizationOperations.subtract.convertAttributes) {
                rt.insert(attribute);
            }

            subtractConst = subtractConstConvert;
        }

        Output<Node> leftBranchParent = dequantizationOperations.subtract.constantIndex == 1 ? parent : subtractConst;
        Output<Node> rightBranchParent = dequantizationOperations.subtract.constantIndex == 1 ? subtractConst : parent;

        if (((dequantizationOperations.subtract.outPrecision == element::undefined) ||
            (dequantizationOperations.subtract.outPrecision == parent.get_element_type())) &&
            (((dequantizationOperations.subtract.constantPrecision == element::undefined) ||
            (dequantizationOperations.subtract.constantPrecision == parent.get_element_type())) ||
            dequantizationOperations.subtract.addConvert)) {
            subtract = dequantizationOperations.subtract.constantIndex == 1ul ?
                std::make_shared<ov::opset1::Subtract>(parent, subtractConst) :
                subtract = std::make_shared<ov::opset1::Subtract>(subtractConst, parent);
        } else {
            if (dequantizationOperations.subtract.constantIndex == 1ul) {
                subtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ element::f32 },
                    ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(subtractConst, element::f32).get());
            } else {
                subtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ element::f32 },
                    ov::op::TemporaryReplaceOutputType(subtractConst, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(parent, element::f32).get());
            }

            ov::pass::low_precision::NetworkHelper::setOutDataPrecision(subtract, dequantizationOperations.subtract.outPrecision);
        }

        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), subtract }, subtract);
        subtract->set_friendly_name(data.get_node_shared_ptr()->get_friendly_name() + "/DequantizationSubtract");

        if (!dequantizationOperations.subtract.attributes.empty()) {
            auto& rt = subtract->get_rt_info();
            for (const auto& attribute : dequantizationOperations.subtract.attributes) {
                rt.insert(attribute);
            }
        }

        parent = subtract;
    }

    if (!dequantizationOperations.multiply.empty()) {
        auto const newMultiply = makeMultiply(parent, dequantizationOperations.multiply);
        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), newMultiply }, newMultiply);
        newMultiply->set_friendly_name(data.get_node_shared_ptr()->get_friendly_name() + "/DequantizationMultiply");
        parent = newMultiply;
    }

    return parent.get_node_shared_ptr();
}

std::shared_ptr<Node> makeMultiply(const Output<Node>& parent, const DequantizationOperations::Multiply& multiply) {
    std::vector<size_t> shape;
    auto values = multiply.values;
    if (multiply.constantShapeIsDefined) {
        shape = multiply.constantShape;
        if (values.size() == 1ul) {
            values = std::vector<float>(shape_size(shape), values[0]);
        }
    } else {
        if (values.size() == 1ul) {
            shape = std::vector<size_t>({});
        } else {
            const auto rank = parent.get_partial_shape().rank();
            shape = std::vector<size_t>(rank.is_dynamic() ? 4ul : rank.get_length(), 1ul);
            shape[shape.size() >= 2 ? 1ul : 0] = values.size();
        }
    }

    std::shared_ptr<ov::opset1::Multiply> newMultiply;
    if (((multiply.outPrecision == element::undefined) ||
        (multiply.outPrecision == parent.get_element_type())) &&
        ((multiply.constantPrecision == element::undefined) ||
        (multiply.constantPrecision == parent.get_element_type()))) {
        const std::shared_ptr<ov::opset1::Constant> constant = std::make_shared<ov::opset1::Constant>(
            multiply.constantPrecision != element::undefined ?
                multiply.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        newMultiply = multiply.constantIndex == 1ul ?
            std::make_shared<ov::opset1::Multiply>(parent, constant) :
            std::make_shared<ov::opset1::Multiply>(constant, parent);
    } else {
        const std::shared_ptr<ov::opset1::Constant> constant = std::make_shared<ov::opset1::Constant>(
            multiply.constantPrecision != element::undefined ?
                multiply.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        // TODO: use templates
        newMultiply = multiply.constantIndex == 1ul ?
            std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ multiply.outPrecision },
                ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(constant, element::f32).get()) :
            std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ multiply.outPrecision },
                ov::op::TemporaryReplaceOutputType(constant, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(parent, element::f32).get());
    }

    return newMultiply;
}

std::shared_ptr<Node> makeReshape(const Output<Node>& data, const Reshape& reshape) {
    auto constant = makeConstant(ov::element::i64, Shape({ reshape.values.size() }), reshape.values);
    return std::make_shared<ov::opset1::Reshape>(data, constant->output(0), reshape.special_zero);
}

std::shared_ptr<Node> makeTranspose(const Output<Node>& data, const Transpose& transpose) {
    auto constant = makeConstant(ov::element::i64, Shape({ transpose.values.size() }), transpose.values);
    return std::make_shared<ov::opset1::Transpose>(data, constant->output(0));
}

std::shared_ptr<ov::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& output,
    const ov::element::Type constantType,
    const FakeQuantizeOnData& fqOnData) {
    return ov::as_type_ptr<ov::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        output,
        constantType,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues));
}

std::shared_ptr<ov::opset1::Convolution> makeConvolution(const Output<Node>& output, const Convolution& convolution) {
    auto parentOnActivations = output;
    if (!convolution.zeroPointOnActivations.empty()) {
        auto constant = std::make_shared<ov::opset1::Constant>(
            convolution.zeroPointOnActivations.outPrecision,
            convolution.zeroPointOnActivations.constantShape,
            convolution.zeroPointOnActivations.values);
        parentOnActivations = std::make_shared<ov::opset1::Subtract>(parentOnActivations, constant);
    }

    assert(!convolution.constantOnWeights.empty());

    ov::Output<ov::Node> weights = std::make_shared<ov::opset1::Constant>(
        convolution.constantOnWeights.outPrecision,
        convolution.constantOnWeights.shape,
        convolution.constantOnWeights.values);

    if (!convolution.dequantizationOnWeights.empty()) {
        weights = makeDequantization(weights, convolution.dequantizationOnWeights);
    }

    return std::make_shared<ov::opset1::Convolution>(
        parentOnActivations,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
}

std::shared_ptr<ov::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const Output<ov::Node>& output,
    const ov::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<ov::opset1::FakeQuantize> fq = makeFakeQuantize(output, precision, fqOnData);
    return std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
        *fq,
        fqOnData.outputPrecision == element::undefined ? precision : fqOnData.outputPrecision);
}

std::shared_ptr<ov::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& input,
    const ov::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const bool subgraphOnConstantPath) {
    std::shared_ptr<Node> inputLowNode;
    std::shared_ptr<Node> inputHighNode;

    if (subgraphOnConstantPath) {
        const auto topConstant = ngraph::builder::makeConstant(constantPrecision, ov::Shape{1}, std::vector<float>(1, 0.f), false);
        const auto convert = std::make_shared<ov::opset1::Convert>(topConstant, element::f32);

        const auto subtractMin = std::make_shared<ov::opset1::Subtract>(
            std::make_shared<ov::opset1::Constant>(constantPrecision, ov::Shape{ 1 }, std::vector<float>{fqOnData.outputLowValues[0]}),
            convert);
        const auto subtractMax = std::make_shared<ov::opset1::Subtract>(
            std::make_shared<ov::opset1::Constant>(constantPrecision, ov::Shape{ 1 }, std::vector<float>{fqOnData.outputHighValues[0]}),
            convert);

        inputLowNode = std::make_shared<ov::opset1::Multiply>(
            std::make_shared<ov::opset1::Constant>(
                constantPrecision,
                ov::Shape{ 1 },
                std::vector<float>{fqOnData.inputLowValues[0] / fqOnData.outputLowValues[0]}),
            subtractMin);
        inputHighNode = std::make_shared<ov::opset1::Multiply>(
            std::make_shared<ov::opset1::Constant>(
                constantPrecision,
                ov::Shape{ 1 },
                std::vector<float>{fqOnData.inputHighValues[0] / fqOnData.outputHighValues[0]}),
            subtractMax);
    } else {
        inputLowNode = ngraph::builder::makeConstant(
            constantPrecision,
            fqOnData.constantShapes.empty() ? ov::Shape{} : fqOnData.constantShapes[0],
            fqOnData.inputLowValues,
            fqOnData.inputLowValues.empty());
        if (fqOnData.addConverts) {
            inputLowNode = std::make_shared<ov::op::v0::Convert>(inputLowNode, ov::element::f32);
        }

        inputHighNode = ngraph::builder::makeConstant(
            constantPrecision,
            fqOnData.constantShapes.empty() ?
                ov::Shape{} :
                (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[1]),
            fqOnData.inputHighValues,
            fqOnData.inputHighValues.empty());
        if (fqOnData.addConverts) {
            inputHighNode = std::make_shared<ov::op::v0::Convert>(inputHighNode, ov::element::f32);
        }
    }

    auto outputLowNode = ngraph::builder::makeConstant(
        constantPrecision,
        fqOnData.constantShapes.empty() ?
            ov::Shape{} :
            (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[2]),
        fqOnData.outputLowValues,
        fqOnData.outputLowValues.empty());
    if (fqOnData.addConverts) {
        outputLowNode = std::make_shared<ov::op::v0::Convert>(outputLowNode, ov::element::f32);
    }

    auto outputHighNode = ngraph::builder::makeConstant(
        constantPrecision,
        fqOnData.constantShapes.empty() ?
            ov::Shape{} :
            (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[3]),
        fqOnData.outputHighValues,
        fqOnData.outputHighValues.empty());
    if (fqOnData.addConverts) {
        outputHighNode = std::make_shared<ov::op::v0::Convert>(outputHighNode, ov::element::f32);
    }

    auto fq = std::make_shared<ov::opset1::FakeQuantize>(input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fqOnData.quantizationLevel);

    auto& rt = fq->get_rt_info();
    for (auto& attribute : fqOnData.attributes) {
        if (attribute.is<ov::RuntimeAttribute>()) {
            rt[attribute.as<ov::RuntimeAttribute>().get_type_info()] = attribute;
        }
    }

    return fq;
}

std::shared_ptr<ov::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ov::Node>& input,
    const ov::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData) {
    const std::shared_ptr<ov::opset1::FakeQuantize> fq = makeFakeQuantize(input, constantPrecision, fqOnData);
    return std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
        *fq,
        fqOnData.outputPrecision == ov::element::undefined ? constantPrecision : fqOnData.outputPrecision);
}

void addAttributes(std::vector<std::shared_ptr<ov::Node>> nodes, std::vector<ov::Any> attributes) {
    for (const auto& node : nodes) {
        for (const auto& attribute : attributes) {
            if (attribute.is<ov::RuntimeAttribute>()) {
                node->get_rt_info()[attribute.as<ov::RuntimeAttribute>().get_type_info()] = attribute;
            }
        }
    }
}

std::shared_ptr<Node> makeConvolution(
    const std::shared_ptr<Node>& parent,
    const element::Type precision,
    const bool weightsWithoutFQ,
    const element::Type weightsprecision) {
    const size_t outputChannels = parent->get_output_partial_shape(0)[1].get_length() * 2;
    const size_t inputChannels = parent->get_output_partial_shape(0)[1].get_length();
    const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };

    std::shared_ptr<Node> weights;
    if (weightsWithoutFQ) {
        weights = std::make_shared<ov::opset1::Constant>(weightsprecision, shape, std::vector<int>(ov::shape_size(shape), 100));
    } else {
        weights = ngraph::builder::makeFakeQuantize(
            std::make_shared<ov::opset1::Constant>(precision, shape, std::vector<float>(ov::shape_size(shape), 1.f)),
            precision,
            255,
            { outputChannels, 1, 1, 1 },
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f),
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f));
        weights->set_friendly_name("fakeQuantizeOnWeights");
    }

    const auto convolution = std::make_shared<ov::opset1::Convolution>(
        ov::op::TemporaryReplaceOutputType(parent, precision).get(),
        ov::op::TemporaryReplaceOutputType(weights, precision).get(),
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });

    convolution->set_friendly_name("convolution");

    return convolution;
}

} // namespace subgraph
} // namespace builder
} // namespace ngraph
