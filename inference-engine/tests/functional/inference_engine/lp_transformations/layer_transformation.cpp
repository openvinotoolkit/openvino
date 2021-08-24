// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <low_precision/network_helper.hpp>
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph::pass;

TestTransformationParams::TestTransformationParams(
    bool updatePrecisions,
    std::vector<element::Type> precisionsOnActivations,
    std::vector<element::Type> precisionsOnWeights,
    bool supportAsymmetricQuantization,
    element::Type deqPrecision,
    bool support3DTensorOnActivations,
    bool deconvolutionSpecificChannelsRatio) :
    updatePrecisions(updatePrecisions),
    precisionsOnActivations(precisionsOnActivations),
    precisionsOnWeights(precisionsOnWeights),
    supportAsymmetricQuantization(supportAsymmetricQuantization),
    deqPrecision(deqPrecision),
    support3DTensorOnActivations(support3DTensorOnActivations),
    deconvolutionSpecificChannelsRatio(deconvolutionSpecificChannelsRatio) {
    if (precisionsOnActivations.size() == 0ul) {
        THROW_TRANSFORMATION_EXCEPTION << "precisions on activations are not specisifed";
    }

    if (precisionsOnWeights.size() == 0ul) {
        THROW_TRANSFORMATION_EXCEPTION << "precisions on weights are not specisifed";
    }
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

TestTransformationParams& TestTransformationParams::setSupport3DTensorOnActivations(const bool support3DTensorOnActivations) {
    this->support3DTensorOnActivations = support3DTensorOnActivations;
    return *this;
}

TestTransformationParams& TestTransformationParams::setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio) {
    this->deconvolutionSpecificChannelsRatio = deconvolutionSpecificChannelsRatio;
    return *this;
}

TestTransformationParams LayerTransformation::createParamsU8U8() {
    return TestTransformationParams(true, { ngraph::element::u8 }, { ngraph::element::u8 });
}

TestTransformationParams LayerTransformation::createParamsU8I8() {
    return TestTransformationParams(true, { ngraph::element::u8 }, { ngraph::element::i8 });
}

TestTransformationParams LayerTransformation::createParamsI8I8() {
    return TestTransformationParams(true, { ngraph::element::i8 }, { ngraph::element::i8 });
}

TestTransformationParams LayerTransformation::createParamsU8I8AndI8() {
    return TestTransformationParams(true, { ngraph::element::u8, ngraph::element::i8 }, { ngraph::element::i8 });
}

pass::low_precision::LayerTransformation::Params TestTransformationParams::toParams(const TestTransformationParams& params) {
    return low_precision::LayerTransformation::Params(
        params.updatePrecisions,
        params.deqPrecision);
}

//TestTransformationParams LayerTransformation::createParamsU8U8() {
//    return low_precision::LayerTransformation::Params(
//        true,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
//        true,
//        { ngraph::element::u8 },
//        { ngraph::element::u8 });
//}
//
//TestTransformationParams LayerTransformation::createParamsU8I8() {
//    return low_precision::LayerTransformation::Params(
//        true,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
//        true,
//        { ngraph::element::u8 },
//        { ngraph::element::i8 });
//}
//
//TestTransformationParams LayerTransformation::createParamsI8I8() {
//    return low_precision::LayerTransformation::Params(
//        true,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
//        true,
//        { ngraph::element::i8 },
//        { ngraph::element::i8 });
//}
//
//TestTransformationParams LayerTransformation::createParamsU8I8AndI8() {
//    return low_precision::LayerTransformation::Params(
//        true,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
//        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
//        true,
//        { ngraph::element::u8, ngraph::element::i8 },
//        { ngraph::element::i8 });
//}

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
    const ngraph::element::Type& type,
    const ngraph::PartialShape& shape,
    const TestTransformationParams& params) {
    std::ostringstream result;
    result << type << "_" << shape << "_" << toString(params);
    return result.str();
}

ngraph::builder::subgraph::DequantizationOperations LayerTransformation::toDequantizationOperations(
    const ngraph::pass::low_precision::FakeQuantizeDequantization& dequantization) {
    const auto convert = dequantization.convert != nullptr ?
        ngraph::builder::subgraph::DequantizationOperations::Convert(dequantization.convert->output(0).get_element_type()) :
        ngraph::builder::subgraph::DequantizationOperations::Convert();

    ngraph::builder::subgraph::DequantizationOperations::Subtract subtract;
    {
        const bool addDequantizationAttribute = dequantization.subtract != nullptr ?
            dequantization.subtract->get_rt_info().count("DEQUANTIZATION") != 0 :
            true;

        const size_t constantIndex = dequantization.subtractConstant && dequantization.subtract ?
            ngraph::pass::low_precision::NetworkHelper::getChildInputIndex(
                dequantization.subtractConvert ? std::dynamic_pointer_cast<ngraph::Node>(dequantization.subtractConvert) : dequantization.subtractConstant,
                dequantization.subtract) :
            0ul;

        subtract = dequantization.subtractConstant != nullptr ?
            ngraph::builder::subgraph::DequantizationOperations::Subtract(
                dequantization.subtractConstant->cast_vector<float>(),
                dequantization.subtract->output(0).get_element_type(),
                dequantization.subtractConstant->output(0).get_shape(),
                addDequantizationAttribute,
                constantIndex,
                dequantization.subtractConstant->output(0).get_element_type(),
                dequantization.subtractConvert != nullptr) :
            ngraph::builder::subgraph::DequantizationOperations::Subtract();
    }

    ngraph::builder::subgraph::DequantizationOperations::Multiply multiply;
    {
        const bool addDequantizationAttribute = dequantization.multiply != nullptr ?
            dequantization.multiply->get_rt_info().count("DEQUANTIZATION") != 0 :
            true;

        const size_t constantIndex = dequantization.multiplyConstant && dequantization.multiply ?
            ngraph::pass::low_precision::NetworkHelper::getChildInputIndex(dequantization.multiplyConstant, dequantization.multiply) :
            0ul;

        multiply = dequantization.multiplyConstant != nullptr ?
            ngraph::builder::subgraph::DequantizationOperations::Multiply(
                dequantization.multiplyConstant->cast_vector<float>(),
                dequantization.multiplyConstant->output(0).get_element_type(),
                dequantization.multiplyConstant->output(0).get_shape(),
                addDequantizationAttribute,
                constantIndex) :
            ngraph::builder::subgraph::DequantizationOperations::Multiply();
    }

    return ngraph::builder::subgraph::DequantizationOperations(convert, subtract, multiply);
}
