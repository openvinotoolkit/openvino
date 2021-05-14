// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include <legacy/net_pass.h>
#include <legacy/graph_transformer.h>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include "legacy/ngraph_ops/fully_connected.hpp"
#include <ngraph/op/gelu.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/pass/convert_prc.hpp"

#include "common_test_utils/common_utils.hpp"
#include <legacy/ie_util_internal.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <low_precision/transformer.hpp>
#include <low_precision/convolution.hpp>

namespace LayerTestsUtils {


ngraph::pass::low_precision::LowPrecisionTransformations LayerTransformation::getLowPrecisionTransformationsNGraph(
    const ngraph::pass::low_precision::LayerTransformation::Params& params) const {
    return ngraph::pass::low_precision::LowPrecisionTransformer::getAllTransformations(params).
        add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(
            ngraph::pass::low_precision::LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 }));
    // addCleanup<ScaleShiftToConvolutionTransformation>(
    //    LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 }),
    //    "ScaleShift"));
}

InferenceEngine::CNNNetwork convert(std::shared_ptr<ngraph::Function> function) {
    InferenceEngine::CNNNetwork net1(function);
    InferenceEngine::CNNNetwork clonedNetwork = InferenceEngine::cloneNetwork(net1);
    if (clonedNetwork.getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
            if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
                return dtsOp->input_value(0).get_shape().size() <= 5lu && dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
            }

            // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
            if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
                return stdOp->input_value(0).get_shape().size() <= 5lu && stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
            }

            if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
                return fc_op->input_value(0).get_shape().size() == 3ul;
            }

            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset2::SpaceToBatch>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node);
        };
        auto nGraphFunc = clonedNetwork.getFunction();

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
        manager.register_pass<ngraph::pass::ConvertPriorBox>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        NGRAPH_SUPPRESS_DEPRECATED_START
        manager.set_callback(transformations_callback);
        NGRAPH_SUPPRESS_DEPRECATED_END
        manager.run_passes(nGraphFunc);
    }

    return clonedNetwork;
}

std::shared_ptr<ngraph::Function> LayerTransformation::transformNGraph(
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::pass::low_precision::LowPrecisionTransformations& transformations) {
    InferenceEngine::CNNNetwork clonedNetwork = convert(function);
    auto nGraphFunc = clonedNetwork.getFunction();

    ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
    transformer.transform(nGraphFunc);

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
        // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
            return dtsOp->input_value(0).get_shape().size() <= 5lu && dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
        }

        // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
        if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
            return stdOp->input_value(0).get_shape().size() <= 5lu && stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
        }

        if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
            return fc_op->input_value(0).get_shape().size() == 3ul;
        }

        if (auto add_op = std::dynamic_pointer_cast<const ngraph::opset1::Add>(node)) {
            return ngraph::is_type<ngraph::opset1::Convolution>(add_op->get_input_node_shared_ptr(0)) ||
                ngraph::is_type<ngraph::opset1::GroupConvolution>(add_op->get_input_node_shared_ptr(0)) ||
                ngraph::is_type<ngraph::opset1::MatMul>(add_op->get_input_node_shared_ptr(0));
        }

        return std::dynamic_pointer_cast<const ngraph::opset2::Gelu>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset2::BatchToSpace>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset2::SpaceToBatch>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset3::ExtractImagePatches>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::HSwish>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::ReduceL1>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::ReduceL2>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::SoftPlus>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    NGRAPH_SUPPRESS_DEPRECATED_START
    manager.set_callback(transformations_callback);
    NGRAPH_SUPPRESS_DEPRECATED_END
    manager.run_passes(nGraphFunc);

    return clonedNetwork.getFunction();
}

InferenceEngine::Precision LayerTransformation::getDeviceInternalPrecision(const InferenceEngine::Precision precision) {
    if (precision == InferenceEngine::Precision::FP16) {
        return InferenceEngine::Precision::FP32;
    }

    return precision;
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParams() {
    return ngraph::pass::low_precision::LayerTransformation::Params(
        true,
        ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true);
}

}  // namespace LayerTestsUtils
