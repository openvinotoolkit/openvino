// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include "generic_ie.hpp"
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>

#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph::pass;

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        true,
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { ngraph::element::u8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsI8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        true,
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { ngraph::element::i8 },
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
    // std::vector<std::shared_ptr<ngraph::Function>> originalModule{ actualFunction };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(originalModule);

    // TODO: refactor: do you really need anything from here?
    //{
    //    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
    //        // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    //        if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
    //            return dtsOp->input_value(0).get_shape().size() <= 5lu && dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
    //        }

    //        // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
    //        if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
    //            return stdOp->input_value(0).get_shape().size() <= 5lu && stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
    //        }

    //        return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
    //            std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node) ||
    //            std::dynamic_pointer_cast<const ::ngraph::opset2::SpaceToBatch>(node) ||
    //            std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node);
    //    };

    //    // Disable shape inference (WA for generic operations)
    //    ::ngraph::op::GenericIE::DisableReshape noReshape(function);

    //    // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
    //    ngraph::pass::CommonOptimizations(transformations_callback).run_on_function(function);
    //    ngraph::pass::ConvertOpSet3ToOpSet2(transformations_callback).run_on_function(function);
    //    ngraph::pass::ConvertOpSet2ToOpSet1(transformations_callback).run_on_function(function);
    //    ngraph::pass::ConvertOpSet1ToLegacy(transformations_callback).run_on_function(function);
    //}

    ngraph::pass::low_precision::LowPrecisionTransformations transformations = ngraph::pass::low_precision::LowPrecisionTransformer::getAllTransformations();
    ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
    transformer.transform(function);

    // std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ actualFunction };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type& type,
    const ngraph::Shape& shape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << type << "_" << shape << "_" << toString(params);
    return result.str();
}
