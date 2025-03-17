// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include "openvino/opsets/opset1.hpp"
#include "low_precision/network_helper.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov::pass;

TestTransformationParams::TestTransformationParams(
    bool updatePrecisions,
    std::vector<element::Type> precisionsOnActivations,
    std::vector<element::Type> precisionsOnWeights,
    bool supportAsymmetricQuantization,
    element::Type deqPrecision,
    bool deconvolutionSpecificChannelsRatio,
    const std::vector<ov::element::Type> defaultPrecisions) :
    updatePrecisions(updatePrecisions),
    precisionsOnActivations(precisionsOnActivations),
    precisionsOnWeights(precisionsOnWeights),
    supportAsymmetricQuantization(supportAsymmetricQuantization),
    deqPrecision(deqPrecision),
    deconvolutionSpecificChannelsRatio(deconvolutionSpecificChannelsRatio),
    defaultPrecisions(defaultPrecisions) {
    if (precisionsOnActivations.size() == 0ul) {
        THROW_TRANSFORMATION_EXCEPTION << "precisions on activations are not specisifed";
    }

    if (precisionsOnWeights.size() == 0ul) {
        THROW_TRANSFORMATION_EXCEPTION << "precisions on weights are not specisifed";
    }
}

TestTransformationParams& TestTransformationParams::setDeqPrecision(const element::Type deqPrecision) {
    this->deqPrecision = deqPrecision;
    return *this;
}

TestTransformationParams& TestTransformationParams::setUpdatePrecisions(const bool updatePrecisions) {
    this->updatePrecisions = updatePrecisions;
    return *this;
}

TestTransformationParams& TestTransformationParams::setSupportAsymmetricQuantization(const bool supportAsymmetricQuantization) {
    this->supportAsymmetricQuantization = supportAsymmetricQuantization;
    return *this;
}

TestTransformationParams& TestTransformationParams::setPrecisionsOnActivations(const std::vector<element::Type>& precisionsOnActivations) {
    this->precisionsOnActivations = precisionsOnActivations;
    return *this;
}

TestTransformationParams& TestTransformationParams::setPrecisionsOnWeights(const std::vector<element::Type>& precisionsOnWeights) {
    this->precisionsOnWeights = precisionsOnWeights;
    return *this;
}

TestTransformationParams& TestTransformationParams::setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio) {
    this->deconvolutionSpecificChannelsRatio = deconvolutionSpecificChannelsRatio;
    return *this;
}

TestTransformationParams& TestTransformationParams::setDefaultPrecisions(const std::vector<element::Type>& defaultPrecisions) {
    this->defaultPrecisions = defaultPrecisions;
    return *this;
}

TestTransformationParams LayerTransformation::createParamsU8U8() {
    return TestTransformationParams(true, { ov::element::u8 }, { ov::element::u8 });
}

TestTransformationParams LayerTransformation::createParamsU8I8() {
    return TestTransformationParams(true, { ov::element::u8 }, { ov::element::i8 });
}

TestTransformationParams LayerTransformation::createParamsI8I8() {
    return TestTransformationParams(true, { ov::element::i8 }, { ov::element::i8 });
}

TestTransformationParams LayerTransformation::createParamsU8I8AndI8() {
    return TestTransformationParams(true, { ov::element::u8, ov::element::i8 }, { ov::element::i8 });
}

ov::pass::low_precision::LayerTransformation::Params TestTransformationParams::toParams(const TestTransformationParams& params) {
    return ov::pass::low_precision::LayerTransformation::Params(
        params.updatePrecisions,
        params.deqPrecision,
        params.defaultPrecisions);
}

std::string LayerTransformation::toString(const TestTransformationParams& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric_" : "symmetric_") <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.precisionsOnActivations[0] << "_" <<
        params.precisionsOnWeights[0];

    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ov::element::Type& type,
    const ov::PartialShape& shape,
    const TestTransformationParams& params) {
    std::ostringstream result;
    result << type << "_" << shape << "_" << toString(params);
    return result.str();
}

ov::builder::subgraph::DequantizationOperations LayerTransformation::toDequantizationOperations(
    const ov::pass::low_precision::FakeQuantizeDequantization& dequantization) {
    const auto convert = dequantization.convert != nullptr ?
        ov::builder::subgraph::DequantizationOperations::Convert(dequantization.convert->output(0).get_element_type()) :
        ov::builder::subgraph::DequantizationOperations::Convert();

    ov::builder::subgraph::DequantizationOperations::Subtract subtract;
    {
        const bool addDequantizationAttribute = dequantization.subtract != nullptr ?
            dequantization.subtract->get_rt_info().count("DEQUANTIZATION") != 0 :
            true;

        const size_t constantIndex = dequantization.subtractConstant && dequantization.subtract ?
            ov::pass::low_precision::NetworkHelper::getChildInputIndex(
                dequantization.subtractConvert ? std::dynamic_pointer_cast<ov::Node>(dequantization.subtractConvert) : dequantization.subtractConstant,
                dequantization.subtract) :
            0ul;

        subtract = dequantization.subtractConstant != nullptr ?
            ov::builder::subgraph::DequantizationOperations::Subtract(
                dequantization.subtractConstant->cast_vector<float>(),
                dequantization.subtract->output(0).get_element_type(),
                dequantization.subtractConstant->output(0).get_shape(),
                addDequantizationAttribute,
                constantIndex,
                dequantization.subtractConstant->output(0).get_element_type(),
                dequantization.subtractConvert != nullptr) :
            ov::builder::subgraph::DequantizationOperations::Subtract();
    }

    ov::builder::subgraph::DequantizationOperations::Multiply multiply;
    {
        const size_t constantIndex = dequantization.multiplyConstant && dequantization.multiply ?
            ov::pass::low_precision::NetworkHelper::getChildInputIndex(dequantization.multiplyConstant, dequantization.multiply) :
            0ul;

        multiply = dequantization.multiplyConstant != nullptr ?
            ov::builder::subgraph::DequantizationOperations::Multiply(
                dequantization.multiplyConstant->cast_vector<float>(),
                dequantization.multiplyConstant->output(0).get_element_type(),
                dequantization.multiplyConstant->output(0).get_shape(),
                false,
                constantIndex) :
            ov::builder::subgraph::DequantizationOperations::Multiply();
    }

    return ov::builder::subgraph::DequantizationOperations(convert, subtract, multiply);
}

bool LayerTransformation::allNamesAreUnique(const std::shared_ptr<ov::Model>& model) {
    const auto& ops = model->get_ops();
    std::set<std::string> opNames;
    for (const auto& op : ops) {
        auto it = opNames.find(op->get_friendly_name());
        if (it != opNames.end()) {
            return false;
        }

        opNames.insert(op->get_friendly_name());
    }

    return true;
}
