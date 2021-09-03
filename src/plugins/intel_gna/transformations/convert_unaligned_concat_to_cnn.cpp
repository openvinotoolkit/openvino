// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_unaligned_concat_to_cnn.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <gna_plugin_log.hpp>

#include <ngraph/shape.hpp>
#include "layers/gna_concat_layer.hpp"
#include <transformations/rt_info/disable_constant_folding.hpp>
#include "backend/gna_limitations.hpp"
#include "gna_lib_ver_selector.hpp"

using namespace GNAPluginNS;
using ngraph::op::Op;
using ngraph::Output;

using GNAPluginNS::GNALimitations::convMinFiltersNum;
using GNAPluginNS::GNALimitations::convFilterSizeDivider;
using GNAPluginNS::GNALimitations::noOfInputsDivisor;
using GNAPluginNS::GNALimitations::gnaTensorDataAlignmentElements;

typedef std::vector<float> DelayedCopyCnnFilter;
typedef std::vector<float> DelayedCopyAffineFilter;

// Example of returned vector
// numberOfFilters = 4             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
// filterLength = 16               0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
// delayedCopySourceOffset = 5     0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
//                                 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
static DelayedCopyCnnFilter getDelayedCopyCnnFilter(const size_t numberOfFilters,
                                             const size_t filterLength,
                                             const size_t delayedCopySourceOffset) {
    IE_ASSERT(filterLength >= numberOfFilters);
    IE_ASSERT(filterLength - numberOfFilters >= delayedCopySourceOffset);
    const auto totalFilterLength = numberOfFilters * filterLength;
    DelayedCopyCnnFilter value(totalFilterLength, 0.f);
    for (uint32_t i = 0; i < numberOfFilters; i++) {
        value.at(i * filterLength + delayedCopySourceOffset + i) = 1;
    }
    return value;
}

// Example of returned vector
// numberOfInputs = 8          0 0 0 0 0 0 0 0
// numberOfOutputs = 5         0 0 0 0 0 0 0 0
// firstInputIndex = 0         1 0 0 0 0 0 0 0
// firstOutputIndex = 2        0 1 0 0 0 0 0 0
//                             0 0 1 0 0 0 0 0
static DelayedCopyAffineFilter GetCopyAffineFilterValues(size_t numberOfInputs, size_t numberOfOutputs, size_t firstInputIndex, size_t firstOutputIndex) {
    IE_ASSERT(numberOfInputs > 0);
    IE_ASSERT(numberOfOutputs > 0);
    IE_ASSERT(numberOfInputs > firstInputIndex);
    IE_ASSERT(numberOfOutputs > firstOutputIndex);
    DelayedCopyAffineFilter values(numberOfInputs * numberOfOutputs, 0);
    const auto additionalOutputs = std::min(firstInputIndex, firstOutputIndex);
    firstInputIndex -= additionalOutputs;
    firstOutputIndex -= additionalOutputs;
    for (; firstInputIndex < numberOfInputs && firstOutputIndex < numberOfOutputs; firstInputIndex++, firstOutputIndex++) {
        values[firstInputIndex * numberOfOutputs + firstOutputIndex] = 1;
    }
    return values;
}

template <typename T, typename N>
std::shared_ptr < ngraph::opset7::Constant > GetConstant2U(T e1, N e2) {
    const auto values = std::vector<uint32_t>{ static_cast<uint32_t>(e1), static_cast<uint32_t>(e2) };
    return std::make_shared< ngraph::opset7::Constant>(ov::element::u32, ngraph::Shape{ 2 }, values);
}

template <typename T>
std::shared_ptr < ngraph::opset7::Constant > CreateConstantU64V(const std::vector<T>& values) {
    return ngraph::opset7::Constant::create(ov::element::u64, ngraph::Shape{ values.size() }, values);
}

static std::string rtValue(size_t val) {
    return std::to_string(val);
}

template <typename N, typename K>
std::shared_ptr < ov::Node > AppendStridedSliceOn2DNode(
    std::shared_ptr<ov::Node> node, N dim2Begin, K dim2End) {
    std::vector<int64_t> mask{};
    IE_ASSERT(node->get_output_size() == 1);
    auto shape = node->get_output_shape(0);
    IE_ASSERT(shape.size() == 2);
    constexpr auto expectedDim1 = 1;
    IE_ASSERT(shape[0] == expectedDim1);
    if (dim2Begin == 0 && dim2End == shape[1]) {
        return node;        // don't append crop as not needed
    }
    auto begin = GetConstant2U(0, dim2Begin);
    auto end = GetConstant2U(expectedDim1, dim2End);
    auto outputNode = std::make_shared < ngraph::opset7::StridedSlice >(node, begin, end, mask, mask);
    return outputNode;
}

static std::shared_ptr< ngraph::opset7::MatMul> AppendAffineCopy(std::shared_ptr<ov::Node> node,
    size_t numberOfInputs, size_t numberOfOutputs,
    size_t firstInputIndex, size_t firstOutputIndex) {
    IE_ASSERT(node->get_output_size() == 1);
    IE_ASSERT(node->get_output_shape(0).size() == 2);
    IE_ASSERT(node->get_output_shape(0)[0] == 1);
    IE_ASSERT(node->get_output_shape(0)[1] == numberOfInputs);
    const auto elementType = node->get_output_element_type(0);
    auto values = GetCopyAffineFilterValues(numberOfInputs, numberOfOutputs,
        firstInputIndex, firstOutputIndex);
    auto shape = ngraph::Shape{ numberOfInputs, numberOfOutputs };
    auto filter = std::make_shared< ngraph::opset7::Constant>(elementType, shape, values);
    auto appended = std::make_shared< ngraph::opset7::MatMul>(node, filter);
    return appended;
}

static void DisableCF(std::shared_ptr<ov::Node> node) {
    node->get_rt_info()[ov::pass::DisableConstantFolding::get_type_info_static()];
    // TODO: investigate why ov::disable_constant_folding(node) is not working for some tests including
    // smoke_concat_multi_input/ConcatMultiInput.CompareWithRefConstOnly/IS=(1.3)(1.3)(1.3)_netPRC=FP32_targetDevice=GNA
    // It would work if DisableConstantFolding.is_copyable() would return true
}

// Returns a set of GNA Prmitives for aligned component of a concat
static ngraph::NodeVector GetComponentsForAlignedComponent(const ov::Output<ov::Node>& preToUnalignedOutputBadShape) {
    const auto componentSize = ov::shape_size(preToUnalignedOutputBadShape.get_shape());
    const auto lastAlignedFromTheEnd = DownAlignTo(componentSize, gnaTensorDataAlignmentElements);

    auto reshape_pattern_initial_pre = CreateConstantU64V(std::vector<size_t> {1, componentSize});
    auto initialReshape = std::make_shared<ngraph::opset7::Reshape>(preToUnalignedOutputBadShape, reshape_pattern_initial_pre, false);
    std::shared_ptr<ov::Node> preToUnalignedNodePadded = initialReshape;
    const auto componentSizePaddedToFc = UpAlignTo(componentSize, noOfInputsDivisor);
    if (componentSize != componentSizePaddedToFc) {
        preToUnalignedNodePadded = std::make_shared<ngraph::opset7::Pad>(initialReshape,
            GetConstant2U(0, 0), GetConstant2U(0, componentSizePaddedToFc - componentSize),
            ov::op::PadMode::CONSTANT);
    }
    DisableCF(preToUnalignedNodePadded);

    ngraph::NodeVector allNewComponents;
    // When component is small it will be handeled with affine filter only - no need to use Copy primitive
    if (lastAlignedFromTheEnd != 0) {
        auto copyComponent = AppendStridedSliceOn2DNode(preToUnalignedNodePadded, 0, lastAlignedFromTheEnd);
        allNewComponents.push_back(copyComponent);
    }

    if (lastAlignedFromTheEnd == componentSizePaddedToFc) {
        // GNA copy primitive is sufficient, no need for FC
        return allNewComponents;
    }

    auto inputForFcAlignedTail = AppendStridedSliceOn2DNode(preToUnalignedNodePadded, lastAlignedFromTheEnd, componentSizePaddedToFc);

    auto outputForFcAlignedTail = AppendAffineCopy(inputForFcAlignedTail,
        componentSizePaddedToFc - lastAlignedFromTheEnd, componentSize - lastAlignedFromTheEnd, 0, 0);

    allNewComponents.push_back(outputForFcAlignedTail);
    return allNewComponents;
}

static ngraph::NodeVector GetComponentsForNotAlignedComponent(const ov::Output<ov::Node>& unalignedOutputBadShape,
    const ov::element::Type outputType,
    const size_t putAt) {
    const auto putAtOffsetFromAligned = putAt % gnaTensorDataAlignmentElements;
    if (putAtOffsetFromAligned == 0) {
        // Aligned concat - don't need special handling using GNA primitives
        THROW_GNA_EXCEPTION << "putAtOffsetFromAligned == 0";
    }

    const auto delayedCopySourceOffset = gnaTensorDataAlignmentElements - putAtOffsetFromAligned;
    const auto totalElems = ov::shape_size(unalignedOutputBadShape.get_shape());

    // filter length for GNA Convolution must be padded to convFilterSizeDivider, i.e., 8
    const auto totalFilter = UpAlignTo(delayedCopySourceOffset + convMinFiltersNum, convFilterSizeDivider);
    auto maxCnnCopyIn = DownAlignTo(totalElems, noOfInputsDivisor);

    auto unalignedOutputBadShapeNode = unalignedOutputBadShape.get_node();
    std::string unalignedOutputBadShapeNodeName;
    size_t unalignedOutputBadShapeNodeInputsNumber = 0;
    if (unalignedOutputBadShapeNode != nullptr) {
        unalignedOutputBadShapeNodeName = unalignedOutputBadShapeNode->get_type_name();
        if (unalignedOutputBadShapeNode->get_output_size() == 1) {
            const auto unalignedOutputBadShapeNodeInputs = unalignedOutputBadShapeNode->output(0).get_target_inputs();
            unalignedOutputBadShapeNodeInputsNumber = unalignedOutputBadShapeNodeInputs.size();
        }
    }

    auto reshape_pattern_initial = CreateConstantU64V(std::vector<size_t> {1, totalElems});
    auto initialReshape = std::make_shared<ngraph::opset7::Reshape>(unalignedOutputBadShape, reshape_pattern_initial, false);

    // Example of concat's output
    // (The scale is not kept)
    //
    // Aligned address                                 putAt                         Aligned                   Aligned
    // X               putAtOffsetFromAligned            |                               X                        X
    // X-----------Previous concat component-------------|---delayedCopySourceOffset-----X---effeciveOutElements--------------|--tail-----|
    // X                                                 |            Unaligned concat component ------- totalElems ---                   |
    std::shared_ptr < ov::Node > convolutionReshaped;
    const auto ramainingOut = totalElems > delayedCopySourceOffset ? totalElems - delayedCopySourceOffset : 0u;
    size_t effeciveOutElements = 0;
    if (maxCnnCopyIn >= totalFilter && ramainingOut > 0) {
        auto maxOutElements = maxCnnCopyIn - totalFilter + convMinFiltersNum;

        // CNN input and output can be shorten if Affine filter based tail copy is needed
        // Those outputs can be computed by the Affine filter based tail copy
        size_t notNeededCnnOutputs = 0;
        if (ramainingOut != maxOutElements) {
            notNeededCnnOutputs = DownAlignTo(maxOutElements % gnaTensorDataAlignmentElements, noOfInputsDivisor);
        }
        effeciveOutElements = maxOutElements - notNeededCnnOutputs;
        const auto effectiveInElements = maxCnnCopyIn - notNeededCnnOutputs;

        auto ss_for_cnn_in = AppendStridedSliceOn2DNode(initialReshape, 0, effectiveInElements);

        auto reshape_pattern = CreateConstantU64V(std::vector<size_t> {1, 1, 1, effectiveInElements});
        auto reshapeCocnatInput = std::make_shared<ngraph::opset7::Reshape>(ss_for_cnn_in, reshape_pattern, false);

        auto filterCnn1DNW = getDelayedCopyCnnFilter(convMinFiltersNum, totalFilter, delayedCopySourceOffset);

        auto filters = std::make_shared<ngraph::opset7::Constant>(outputType,
                                                                  ngraph::Shape{convMinFiltersNum, 1, 1, totalFilter},
                                                                  filterCnn1DNW);

        auto convolution = std::make_shared<ngraph::opset7::Convolution>(reshapeCocnatInput,
                                                                         filters,
                                                                         ngraph::Strides{1, convMinFiltersNum},
            ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        auto reshape_pattern2 = GetConstant2U(1, effeciveOutElements);
        convolutionReshaped = std::make_shared<ngraph::opset7::Reshape>(convolution, reshape_pattern2, false);
    }

    const auto minHeadCopyElems = std::min(delayedCopySourceOffset, totalElems);
    const auto numberOfInputsForHeadAffineFilter = UpAlignTo(minHeadCopyElems, noOfInputsDivisor);
    const auto numberOfOutputsForHeadAffineFilter =
        std::min(gnaTensorDataAlignmentElements, minHeadCopyElems + putAtOffsetFromAligned);
    const auto firstInputIndexForHeadAffineFilter = 0u;
    const auto firstOutputIndexForHeadAffineFilter = putAtOffsetFromAligned;

    auto tailNotNeeded = totalElems <= delayedCopySourceOffset + effeciveOutElements;
    const auto numberOfInputsForHeadAffineFilterReduced = std::min(totalElems, numberOfInputsForHeadAffineFilter);
    auto fcHead = AppendStridedSliceOn2DNode(initialReshape, 0, numberOfInputsForHeadAffineFilterReduced);
    if (numberOfInputsForHeadAffineFilter != numberOfInputsForHeadAffineFilterReduced) {
        fcHead = std::make_shared<ngraph::opset7::Pad>(fcHead,
            GetConstant2U(0, 0), GetConstant2U(0, numberOfInputsForHeadAffineFilter - numberOfInputsForHeadAffineFilterReduced),
            ov::op::PadMode::CONSTANT);
        // W/A for test smoke_concat_multi_input/ConcatMultiInput.CompareWithRefConstOnly/IS=(1.3)(1.3)(1.3)_netPRC=FP32_targetDevice=GNA
        if (unalignedOutputBadShapeNodeName == std::string("Constant") && unalignedOutputBadShapeNodeInputsNumber == 1) {
            DisableCF(initialReshape);
        }
    }

    auto headMatmul = AppendAffineCopy(fcHead, numberOfInputsForHeadAffineFilter, numberOfOutputsForHeadAffineFilter,
        firstInputIndexForHeadAffineFilter, firstOutputIndexForHeadAffineFilter);

    // Crop part of MatMul output from head of unaligned component
    // Matmul produces numberOfOutputsForHeadAffineFilter outputs, but only the last delayedCopySourceOffset contain valid data
    auto vSSHead = AppendStridedSliceOn2DNode(headMatmul, firstOutputIndexForHeadAffineFilter, numberOfOutputsForHeadAffineFilter);

    std::shared_ptr < ov::Node > vSSTail;
    if (!tailNotNeeded) {
        const auto outOffsetForFCNotAligned = putAt + delayedCopySourceOffset + effeciveOutElements;
        const auto outOffsetAlignedToPutTail = DownAlignTo(outOffsetForFCNotAligned, gnaTensorDataAlignmentElements);
        const auto outnumberOfTailFCOut = putAt + totalElems - outOffsetAlignedToPutTail;
        const auto minNeededTail = totalElems - delayedCopySourceOffset - effeciveOutElements;
        const auto tailBegin = totalElems - outnumberOfTailFCOut;
        const auto tailbeginAligned = DownAlignTo(tailBegin, gnaTensorDataAlignmentElements);
        const auto realNumberofInputsForTailAffineFilter = totalElems - tailbeginAligned;
        const auto gnaAdjustedNumberofInputsForTailAffineFilter =
            UpAlignTo(realNumberofInputsForTailAffineFilter, noOfInputsDivisor);
        const auto firstInputElement = realNumberofInputsForTailAffineFilter - minNeededTail;
        const auto firstOutputElementTail = outnumberOfTailFCOut - minNeededTail;
        auto realInputForAffineCopyFilterForTail = AppendStridedSliceOn2DNode(initialReshape,
            tailbeginAligned, tailbeginAligned + realNumberofInputsForTailAffineFilter);

        std::shared_ptr<ov::Node> inputForAffineCopyFilterForTail = realInputForAffineCopyFilterForTail;

        // To handle the case where (realNumberofInputsForTailAffineFilter) is not multiple of noOfInputsDivisor (i.e., 8)
        // Pad operation has to be inserted
        if (gnaAdjustedNumberofInputsForTailAffineFilter != realNumberofInputsForTailAffineFilter) {
            inputForAffineCopyFilterForTail = std::make_shared<ngraph::opset7::Pad>(realInputForAffineCopyFilterForTail,
                GetConstant2U(0, 0), GetConstant2U(0, gnaAdjustedNumberofInputsForTailAffineFilter - realNumberofInputsForTailAffineFilter),
                ov::op::PadMode::CONSTANT);
            gnalog() << "Size of the tail of unaligned concat component is not multiple of noOfInputsDivisor=="
                     << noOfInputsDivisor << ", Pad: " << inputForAffineCopyFilterForTail->get_friendly_name()
                     << " inserted\n";
        }

        auto tailMatmul = AppendAffineCopy(inputForAffineCopyFilterForTail, gnaAdjustedNumberofInputsForTailAffineFilter, outnumberOfTailFCOut,
            firstInputElement, firstOutputElementTail);

        vSSTail = AppendStridedSliceOn2DNode(tailMatmul, firstOutputElementTail, outnumberOfTailFCOut);
    }

    ngraph::NodeVector allNewComponents;
    allNewComponents.push_back(vSSHead);
    if (convolutionReshaped) {
        allNewComponents.push_back(convolutionReshaped);
    }
    if (vSSTail) {
        allNewComponents.push_back(vSSTail);
    }
    return allNewComponents;
}
struct EnforceOrderResult {
    std::shared_ptr<ov::Node> subgraph;
    std::shared_ptr<ov::Node> previousNode;
};

// previousNodeExecutedAfter - original concat input which should be decomposed into GNA primitives
// to be executed after the components in subgraph
// Result:
// previousNodeExecutedAfter     subgraph (GNA- friendly network handling (N + 1)'s component of the concat)
//   (N's component of the concat)        / / /
//                \                      / / /
//                 \                    / / /
//               Concat (for ordering purposes only)
//              /                                |
//             /                                 |
//  StridedSlice                            StridedSlice
//      |                                        |
//   effectivelly same as                   effectivelly same as
//   previousNodeExecutedAfter              (N + 1)'s component of the concat
//   subgraph for N's components            goes directly to final Concat
//   would be attached here
static EnforceOrderResult EnforceOrder(const ngraph::NodeVector subgraph , const ov::Output<ov::Node> previousNodeExecutedAfter) {
    const auto previousNodeShape = previousNodeExecutedAfter.get_shape();
    IE_ASSERT(previousNodeShape.size() >= 1);
    IE_ASSERT(previousNodeShape[0] != 0);
    const auto previousNodeSize = ov::shape_size(previousNodeShape);
    auto previousNodeExecutedAfterReshaped = previousNodeExecutedAfter;
    if (previousNodeShape.size() != 2 || previousNodeShape[0] != 1) {
        auto shape1xN = GetConstant2U(1, previousNodeSize);
        previousNodeExecutedAfterReshaped = std::make_shared<ngraph::opset7::Reshape>(previousNodeExecutedAfter, shape1xN, false);
    }
    ov::OutputVector fakeConcatOutputs = { previousNodeExecutedAfterReshaped };
    fakeConcatOutputs.insert(fakeConcatOutputs.end(), subgraph.begin(), subgraph.end());
    // this concat could be detected using LayerInfo.isConcatOrdering()
    std::shared_ptr<ov::Node> newFakeOrderingConcat = std::make_shared < ngraph::opset7::Concat>(fakeConcatOutputs, orderingConcatAxis);

    const auto concatShape = newFakeOrderingConcat->get_shape();
    const auto concatShapeSize = ov::shape_size(concatShape);

    // Each ordering concat has two Strided Slices, the first one takes a part from the beginning
    // and all GNA primitives needed to put them into final concat output will be appended
    auto previousNode = AppendStridedSliceOn2DNode(newFakeOrderingConcat, 0, previousNodeSize);
    // the second Crop takes the remiaining output from ordering concat and
    // pass it directelly to the final concat
    auto followingNode = AppendStridedSliceOn2DNode(newFakeOrderingConcat, previousNodeSize, concatShapeSize);

    return { followingNode, previousNode };
}

static std::vector<size_t> getDataFromConstant(ov::Input<ov::Node> begin) {
    auto out = begin.get_source_output();
    auto node = out.get_node();
    auto constant = dynamic_cast <ngraph::opset7::Constant*>(node);
    if (constant == nullptr) {
        return {};
    }
    auto v = constant->cast_vector<size_t>();
    return v;
}

static bool trivialConcatFromStrides(Output<ov::Node> concatOutput) {
    constexpr size_t supportedShapeSize = 2;
    auto concat = concatOutput.get_node();
    auto ssNode = concat->input(0).get_source_output().get_node();
    if (ssNode == nullptr || ssNode->get_input_size() == 0) {
        return false;
    }
    auto directNodeOutput = ssNode->input(0).get_source_output();

    auto concatInputs = concat->inputs();
    size_t expectedBegin = 0;
    for (auto&& i : concatInputs) {
        auto node = i.get_source_output().get_node();
        auto ss = dynamic_cast <ngraph::opset7::StridedSlice*>(node);
        if (ss == nullptr || ss->get_input_size() != 4) {
            return false;
        }
        auto directNodeOutputThis = ss->input(0).get_source_output();
        if (directNodeOutputThis != directNodeOutput) {
            return false;
        }
        auto begin = ss->input(1);
        auto end = ss->input(2);
        auto stride = ss->input(3);

        auto b = getDataFromConstant(begin);
        auto e = getDataFromConstant(end);
        auto s = getDataFromConstant(stride);
        if (b.size() != supportedShapeSize || e.size() != supportedShapeSize || s.size() != supportedShapeSize) {
            return false;
        }
        if (b[0] != 0 || b[1] != expectedBegin || e[0] != 1 || s[0] != 1 || s[1] != 1) {
            return false;
        }
        expectedBegin = e[1];
    }
    auto directNodeOutputShape = directNodeOutput.get_shape();
    if (directNodeOutputShape.size() != supportedShapeSize || directNodeOutputShape[0] != 1 ||
        directNodeOutputShape[1] != expectedBegin) {
        return false;
    }

    return true;
}

static void ConvertTrivialConcatFromStrides(std::shared_ptr<ov::Node> concat) {
    auto ssNode = concat->input(0).get_source_output().get_node();
    auto directNodeOutput = ssNode->input(0).get_source_output();
    auto orgConcatOutput = concat->output(0);

    const auto orgConcatOutputShape = orgConcatOutput.get_shape();
    const auto orgConcatOutputReShape = CreateConstantU64V(orgConcatOutputShape);
    auto concatReplacement = std::make_shared<ngraph::opset7::Reshape>(directNodeOutput, orgConcatOutputReShape, false);
    concatReplacement->set_friendly_name(concat->get_friendly_name());

    for (auto&& inputOfConcatConsumer : orgConcatOutput.get_target_inputs()) {
        inputOfConcatConsumer.replace_source_output(concatReplacement->output(0));
    }
}

static void ConvertUnalignedConcat(std::shared_ptr<ov::Node> orgConcatNode) {
    constexpr int64_t newConcatAxis = 1;
    ngraph::NodeVector allNewComponents;
    const auto orgConcatOutput = orgConcatNode->output(0);
    const auto concatOutputType = orgConcatOutput.get_element_type();
    const auto inputs = orgConcatNode->inputs();
    auto totalOffset = std::accumulate(inputs.cbegin(), inputs.cend(), static_cast<size_t>(0),
        [](const size_t& sum, const ov::Input<ov::Node>& i) {return sum + ov::shape_size(i.get_shape()); });
    auto outputNode = inputs.rbegin()->get_source_output();

    // The loop to convert concat components into GNA-friendly primitieves goes
    // from the last concat component as the component handling may overwrite
    // some previous components buffer
    for (auto input = inputs.rbegin(); input != inputs.rend(); ++input) {
        const auto sizeCurrent = ov::shape_size(input->get_shape());
        totalOffset -= sizeCurrent;
        ngraph::NodeVector components;

        if (totalOffset % gnaTensorDataAlignmentElements == 0) {
            components = GetComponentsForAlignedComponent(outputNode);
        } else {
            components = GetComponentsForNotAlignedComponent(outputNode, concatOutputType, totalOffset);
        }
        const auto previousInput = std::next(input);
        if (previousInput != inputs.rend()) {
            auto result = EnforceOrder(components, previousInput->get_source_output());
            components = { result.subgraph };
            outputNode = result.previousNode;   // for the next iteration
        }

        allNewComponents.insert(allNewComponents.begin(), components.begin(), components.end());
    }

    auto newConcat = std::make_shared < ngraph::opset7::Concat>(allNewComponents, newConcatAxis);

    const auto originalConcatOutputShape = orgConcatOutput.get_shape();
    const auto originalConcatOutputReShape = CreateConstantU64V(originalConcatOutputShape);
    auto newReshapeOutput = std::make_shared<ngraph::opset7::Reshape>(newConcat, originalConcatOutputReShape, false);

    for (auto&& inputOfConcatConsumer : orgConcatOutput.get_target_inputs()) {
        inputOfConcatConsumer.replace_source_output(newReshapeOutput);
    }
    newReshapeOutput->set_friendly_name(orgConcatNode->get_friendly_name());
}

static std::function<bool(Output<ov::Node>)> CheckUnalignedConcat() {
    return [](Output<ov::Node> output) -> bool {
        auto concat_node = output.get_node();
        auto concat = dynamic_cast<ngraph::opset7::Concat*>(concat_node);
        if (concat == nullptr) {
            return false;
        }
        const auto numberOfInputsToConcat = concat->get_input_size();
        if (numberOfInputsToConcat < 2) {
            return false;
        }
        if (concat->get_axis() != 0) {
            auto outputShape = concat->get_output_shape(0);
            outputShape.resize(concat->get_axis());
            if (ov::shape_size(outputShape) > 1) {
                return false;
            }
        }
        size_t offset = 0;
        for (auto&& c : concat->inputs()) {
            if (offset % gnaTensorDataAlignmentElements != 0) {
                return true;
            }
            offset += ov::shape_size(c.get_shape());
        }
        return false;
    };
}

static std::function<bool(Output<ov::Node>)> CheckTrivialStrideConcat() {
    return [](Output<ov::Node> output) -> bool {
        return trivialConcatFromStrides(output);
    };
}

NGRAPH_RTTI_DEFINITION(ConvertUnalignedConcatIntoGnaGraph, "ConvertUnalignedConcatIntoGnaGraph", 0);
ConvertUnalignedConcatIntoGnaGraph::ConvertUnalignedConcatIntoGnaGraph() {
    MATCHER_SCOPE(TransformConcatForGna);
    auto concatPattern = ngraph::pattern::wrap_type<ngraph::opset7::Concat>({}, CheckUnalignedConcat());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = pattern_map.at(concatPattern).get_node_shared_ptr();

        ConvertUnalignedConcat(concat_node);
        unaligned_concat_converted = true;
        return true;
    };

    auto unalignedConcatMatcher = std::make_shared<ngraph::pattern::Matcher>(concatPattern, matcher_name);

    this->register_matcher(unalignedConcatMatcher, callback);
}

NGRAPH_RTTI_DEFINITION(RemoveTrivialStrideConcatPattern, "RemoveTrivialStrideConcatPattern", 0);
RemoveTrivialStrideConcatPattern::RemoveTrivialStrideConcatPattern() {
    MATCHER_SCOPE(RemoveTrivialStrideConcatPattern);
    auto concatPattern = ngraph::pattern::wrap_type<ngraph::opset7::Concat>({}, CheckTrivialStrideConcat());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = pattern_map.at(concatPattern).get_node_shared_ptr();

        ConvertTrivialConcatFromStrides(concat_node);
        return true;
    };

    auto unalignedConcatMatcher = std::make_shared<ngraph::pattern::Matcher>(concatPattern, matcher_name);

    this->register_matcher(unalignedConcatMatcher, callback);
}

NGRAPH_RTTI_DEFINITION(TransformConcatForGna, "TransformConcatForGna", 0);
TransformConcatForGna::TransformConcatForGna() {
    MATCHER_SCOPE(TransformConcatForGna);
    add_matcher<GNAPluginNS::RemoveTrivialStrideConcatPattern>();
    convUCPass = add_matcher<GNAPluginNS::ConvertUnalignedConcatIntoGnaGraph>();
}

bool TransformConcatForGna::unalignedConcatIntoGnaGraphConverted() const {
    return convUCPass->unaligned_concat_converted;
}
