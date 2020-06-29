// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/convolution.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

ConvolutionTransformation::ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {
}

void ConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Convolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>()}));
}

void ConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();

    // Almost all checks that was in WeightableLayerTransformation now should be included into the pattern in registerMatcherIn
    if (!WeightableLayerTransformation::canBeTransformed(context, convolution)) {
        return;
    }

    //// dequantizationOnData
    // std::shared_ptr<Node> dequantizationOperationOnData = convolution->input_value(0).get_node_shared_ptr();
    // FakeQuantizeDequantization dequantizationOnData = getDequantization(*dequantizationOperationOnData);

    //// dequantizationOnWeights
    // std::shared_ptr<Node> parentOnWeights = convolution->input_value(1).get_node_shared_ptr();
    // std::shared_ptr<opset1::FakeQuantize> fakeQuantizeOnWeights = as_type_ptr<opset1::FakeQuantize>(parentOnWeights);
    // const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantizeOnWeights);
    // const DataPrecision dataPrecisionOnWeights = getDataPrecision(
    //    fakeQuantizeOnWeights,
    //    quantizationDetails,
    //    true,
    //    supportAsymmetricQuantization);
    // FakeQuantizeDequantization dequantizationOnWeights = getFakeQuantizeDequantization(
    //    fakeQuantizeOnWeights,
    //    dataPrecisionOnWeights.precision,
    //    dataPrecisionOnWeights.min,
    //    dataPrecisionOnWeights.max);

    std::shared_ptr<Node> dequantizationOperationOnData = convolution->input_value(0).get_node_shared_ptr();

    {
        // Move Multiply from Data path immediately to the output
        // Before moving, Multiply should be decoupled and exchanged with Add in MultiplyAdd operation
        // auto newMultiplyFromData = swapMultiplyAndAdd(
        //        decomposeMultiplyAdd(as_type_ptr<ngraph::op::MultiplyAdd>(scaleShiftOnData)));
        // auto newAdd = optimizeAdd(as_type_ptr<opset1::Add>(newMultiplyFromData->input_value(0).get_node_shared_ptr()));

        auto subtract = as_type_ptr<opset1::Subtract>(dequantizationOperationOnData->input_value(0).get_node_shared_ptr());
        if (subtract != nullptr) {
            subtract = as_type_ptr<opset1::Subtract>(optimizeSubtract(subtract));
        }

        // FIXME: next workaround normalizes shape of newAdd to match CPU plugin expectations
        if (subtract && subtract->get_output_partial_shape(0) != subtract->get_input_partial_shape(1)) {
            size_t length = subtract->get_output_partial_shape(0).rank().get_length();

            // Insert explicit broadcast for channel dimension [1] and immediately fold it
            Shape broadcastShape(subtract->get_output_partial_shape(0).rank().get_length(), 1);
            broadcastShape[1] = subtract->get_output_shape(0)[1];

            std::shared_ptr<Node> newShift = fold<opset1::Broadcast>(
                subtract->input_value(1).get_node_shared_ptr(),
                std::make_shared<opset1::Constant>(
                    element::i64,
                    Shape{ length },
                    broadcastShape));

            replace_node(subtract, subtract->clone_with_new_inputs({ subtract->input_value(0).get_node_shared_ptr(), newShift }));
        }

        std::shared_ptr<ngraph::opset1::Multiply> newMultiplyFromData = as_type_ptr<ngraph::opset1::Multiply>(dequantizationOperationOnData);

        // double-check that Multiply is still scalar-like
        // assert(isScalarLike(as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
        std::shared_ptr<ngraph::opset1::Multiply> newMultiplyAfter = std::make_shared<opset1::Multiply>(
                convolution->copy_with_new_inputs({newMultiplyFromData->input_value(0), convolution->input_value(1)}),
                distillToScalar(as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
        replace_node(convolution, newMultiplyAfter);
        convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();

        if (is_type<opset1::Convert>(convolution->get_input_node_ptr(0))) {
            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_ptr(0)->get_input_node_shared_ptr(0),
                convolution->get_input_node_shared_ptr(1) });
            replace_node(convolution, newConvolution);
            convolution = newConvolution;
        }
    }

    // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    {
        decomposeFakeQuantizeForWeightsPath(convolution, supportAsymmetricQuantization);

        // reassign, because the old one is obsolete and replaced

        // TODO: refactor: operate only with dequantization operations here (no Reshape)

        // TODO: refactor: dequantization operations return from decomposeFakeQuantizeForWeightsPath
        std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(convolution->input_value(1).get_node_shared_ptr());
        std::shared_ptr<opset1::Multiply> multiplyFromWeights = as_type_ptr<opset1::Multiply>(
            reshapeFromWeights == nullptr ?
            convolution->input_value(1).get_node_shared_ptr() :
            convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Subtract> subtractFromWeights = as_type_ptr<opset1::Subtract>(multiplyFromWeights->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Convert> convertFromWeights = as_type_ptr<opset1::Convert>(subtractFromWeights == nullptr ?
            multiplyFromWeights->get_input_node_shared_ptr(0) :
            subtractFromWeights->get_input_node_shared_ptr(0));

        {
            // TODO: temporary workaround
            // if (multiplyFromWeights == nullptr) {
            //    multiplyFromWeights = as_type_ptr<opset1::Multiply>(convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
            // }

            // Check if all dimensions of scale except the first one (which is O-Output channels dimension) are all ones
            auto weightScaleShape = multiplyFromWeights->get_input_shape(1);
            // FIXME: check for rank and 1D-like multiplier
            // if (weightScaleShape.size() <= 2 && shape_size(weightScaleShape) != weightScaleShape[0]) {
            //    // TODO: should we roll back all changes in the network?
            //    return;
            // }

            // It has been just checked that weights scale is effectively 1D tensor, so we can reshape it to [X, 1, ..., 1]
            // to move to the output
            auto newScaleShape = weightScaleShape;
            newScaleShape.pop_back();   // that's all we need: [C, 1, 1, 1] => [C, 1, 1]
            // std::cerr << newScaleShape << "\n";
            // std::cerr << *newMultiplyFromWeights->input_value(1).get_node_shared_ptr();

            // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

            if (reshapeFromWeights != nullptr) {
                reshapeFromWeights = as_type_ptr<opset1::Reshape>(reshapeFromWeights->copy_with_new_inputs({
                    multiplyFromWeights->input_value(0),
                    reshapeFromWeights->input_value(1) }));
            }

            auto newMultiplyAfter = std::make_shared<opset1::Multiply>(
                convolution->copy_with_new_inputs({
                    convolution->input_value(0),
                    reshapeFromWeights != nullptr ?
                        reshapeFromWeights :
                        multiplyFromWeights->input_value(0)
                    }),
                fold_reshape<opset1::Reshape>(
                    multiplyFromWeights->input_value(1),
                    std::make_shared<opset1::Constant>(element::u64, Shape{ newScaleShape.size() }, newScaleShape),
                    false));
            replace_node(convolution, newMultiplyAfter);
            convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        }

        if (subtractFromWeights != nullptr) {
            optimizeSubtract(subtractFromWeights);
        }

        if (convertFromWeights != nullptr) {
            // TODO: why childNode is removed Multiply ???
            // std::shared_ptr<Node> childNode = convertFromWeights->get_output_target_inputs(0).begin()->get_node()->shared_from_this();

            std::shared_ptr<Node> childNode = reshapeFromWeights == nullptr ? convolution : reshapeFromWeights;

            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_shared_ptr(0),
                childNode.get() == convolution.get() ?
                    convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0) :
                    // TODO: hardcoded inputs
                    childNode->copy_with_new_inputs({convertFromWeights->input_value(0), childNode->input_value(1)})});
            replace_node(convolution, newConvolution);
            convolution = newConvolution;
        }
    }

    std::shared_ptr<ngraph::opset1::Multiply> finalDequantization = optimizeMultipliesAfter(
        convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this());

    // TODO: move to base class
    const std::string originalName = convolution->get_friendly_name();
    convolution->set_friendly_name(originalName + "_original");
    finalDequantization->set_friendly_name(originalName);

    // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });


    // std::shared_ptr<Node> newParentOnData;
    // if (dequantizationOnData.subtract != nullptr) {
    //    parent = dequantizationOnData.subtract;
    // } else {
    //    dequantizationOnData.convert
    // }

    // std::shared_ptr<ngraph::opset1::Multiply> newMultiplyAfter = std::make_shared<opset1::Multiply>(
    //    convolution->copy_with_new_inputs({convolution->input_value(0), newMultiplyFromWeights->input_value(0)}),
    //    fold_reshape<opset1::Reshape>(
    //            newMultiplyFromWeights->input_value(1),
    //            std::make_shared<opset1::Constant>(
    //                    element::u64,
    //                    Shape{newScaleShape.size()},
    //                    newScaleShape)->output(0),
    //            false));

    // std::cout << "ConvolutionTransformation::transform: done: " << convolution->get_friendly_name() << std::endl;
}

void ConvolutionTransformation::transform2(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();
    // Almost all checks that was in WeightableLayerTransformation now should be included into the pattern in registerMatcherIn
    if (!WeightableLayerTransformation::canBeTransformed(context, convolution)) {
        return;
    }

    // std::cerr << "Match convolution: " << m.get_match_root()->get_friendly_name() << "\n";

    auto scaleShiftOnData = convolution->input_value(0).get_node_shared_ptr();
    auto parentOnWeights = convolution->input_value(1).get_node_shared_ptr();

    /*
    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(scaleShiftOnData, originalDataDequantizationScales, originalDataDequantizationShifts);

    const bool isDepthwiseConvolution = isDepthwise(layer);
        // Skip checking as it is already checked in WeightableLayerTransformation::canBeTransformed
        // TODO: Acknowledge and remove this block
    }*/

    {
        // Move Multiply from Data path immediately to the output
        // Before moving, Multiply should be decoupled and exchanged with Add in MultiplyAdd operation
        //auto newMultiplyFromData = swapMultiplyAndAdd(
        //        decomposeMultiplyAdd(as_type_ptr<ngraph::op::MultiplyAdd>(scaleShiftOnData)));
        //auto newAdd = optimizeAdd(as_type_ptr<opset1::Add>(newMultiplyFromData->input_value(0).get_node_shared_ptr()));
        //// FIXME: next workaround normalizes shape of newAdd to match CPU plugin expectations
        //if (newAdd && newAdd->get_output_partial_shape(0) != newAdd->get_input_partial_shape(1)) {
        //    // Insert explicit broadcast for channel dimension [1] and immediately fold it
        //    Shape broadcastShape(newAdd->get_output_partial_shape(0).rank().get_length(), 1);
        //    broadcastShape[1] = newAdd->get_output_shape(0)[1];
        //    auto newShift = fold<opset1::Broadcast>(newAdd->input_value(1),
        //            std::make_shared<opset1::Constant>(
        //                    element::i64,
        //                    Shape{size_t(newAdd->get_output_partial_shape(0).rank().get_length())},
        //                    broadcastShape));
        //    replace_node(newAdd->input_value(1).get_node_shared_ptr(), newShift);
        //}

        auto newMultiplyFromData = as_type_ptr<ngraph::opset1::Multiply>(scaleShiftOnData);
        // double-check that Multiply is still scalar-like
        assert(isScalarLike(as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
        auto newMultiplyAfter = std::make_shared<opset1::Multiply>(
                convolution->copy_with_new_inputs({newMultiplyFromData->input_value(0), convolution->input_value(1)}),
                distillToScalar(
                        as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
        replace_node(convolution, newMultiplyAfter);
        convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();
    }

    {
        decomposeFakeQuantizeForWeightsPath(convolution, supportAsymmetricQuantization);

        // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

        // reassign, because the old one is obsolete and replaced
        parentOnWeights = convolution->input_value(1).get_node_shared_ptr();
        //auto newMultiplyFromWeights = swapMultiplyAndAdd(decomposeMultiplyAdd(as_type_ptr<ngraph::op::MultiplyAdd>(parentOnWeights)));
        auto newMultiplyFromWeights = parentOnWeights;

        // Check if all dimensions of scale except the first one (which is O-Output channels dimension) are all ones
        auto weightScaleShape = newMultiplyFromWeights->get_input_shape(1);
        // FIXME: check for rank and 1D-like multiplier
        // if (weightScaleShape.size() <= 2 && shape_size(weightScaleShape) != weightScaleShape[0]) {
        //    // TODO: should we roll back all changes in the network?
        //    return;
        // }

        // It has been just checked that weights scale is effectively 1D tensor, so we can reshape it to [X, 1, ..., 1]
        // to move to the output
        auto newScaleShape = weightScaleShape;
        newScaleShape.pop_back();   // that's all we need: [C, 1, 1, 1] => [C, 1, 1]
        // std::cerr << newScaleShape << "\n";
        // std::cerr << *newMultiplyFromWeights->input_value(1).get_node_shared_ptr();

        auto newMultiplyAfter = std::make_shared<opset1::Multiply>(
                convolution->copy_with_new_inputs({convolution->input_value(0), newMultiplyFromWeights->input_value(0)}),
                fold_reshape<opset1::Reshape>(
                        newMultiplyFromWeights->input_value(1),
                        std::make_shared<opset1::Constant>(
                                element::u64,
                                Shape{newScaleShape.size()},
                                newScaleShape)->output(0),
                        false));
        replace_node(convolution, newMultiplyAfter);
        convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();

        // Handle remaining Add
        auto remainingAdd = as_type_ptr<opset1::Subtract>(convolution->input_value(1).get_node_shared_ptr());
        optimizeSubtract(remainingAdd);
    }

    optimizeMultipliesAfter(convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
}

}// namespace low_precision
}// namespace pass
}// namespace ngraph
