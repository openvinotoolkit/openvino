// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <low_precision/network_helper.hpp>
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph::pass;

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8U8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::u8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsI8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::i8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8AndI8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8, ngraph::element::i8 },
        { ngraph::element::i8 });
}

std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric_" : "symmetric_") <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.precisionsOnActivations[0] << "_" <<
        params.precisionsOnWeights[0] << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

void LayerTransformation::transform(std::shared_ptr<ngraph::Function> function) {
    ngraph::pass::low_precision::LowPrecisionTransformations transformations = ngraph::pass::low_precision::LowPrecisionTransformer::getAllTransformations();
    ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
    transformer.transform(function);
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type& type,
    const ngraph::Shape& shape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
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
