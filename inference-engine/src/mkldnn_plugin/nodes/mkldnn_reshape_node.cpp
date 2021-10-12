// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNReshapeNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ngraph::opset1::Reshape>(op) &&
            !std::dynamic_pointer_cast<const ngraph::opset1::Squeeze>(op) &&
                !std::dynamic_pointer_cast<const ngraph::opset1::Unsqueeze>(op)) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";

    if (isDynamicNode()) {
        auto checkSecondInput = [](const std::shared_ptr<ngraph::Node>& op, const std::string opType) {
            if (op->get_input_partial_shape(1).is_dynamic())
                IE_THROW() << "CPU plug-in doesn't support " << opType << " node with non static second input: " << op->get_friendly_name();
        };

        if (std::dynamic_pointer_cast<const ngraph::opset1::Reshape>(op)) {
            checkSecondInput(op, "Reshape");
        } else if (std::dynamic_pointer_cast<const ngraph::opset1::Squeeze>(op)) {
            if (op->get_input_size() == 1)
                IE_THROW() << "CPU plug-in doesn't support Squeeze node with inputs num equal 1";
            checkSecondInput(op, "Squeeze");
        } else if (std::dynamic_pointer_cast<const ngraph::opset1::Unsqueeze>(op)) {
            checkSecondInput(op, "Unsqueeze");
        } else {
            IE_THROW() << "Unsupported operation type via reshape node";
        }
    }
}

bool MKLDNNReshapeNode::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (lastSecondInputValues.empty())
        return true;
    const int32_t *sndInput = reinterpret_cast<const int32_t *>(getParentEdgesAtPort(1)[0]->getMemory().GetPtr());
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        if (lastSecondInputValues[i] != sndInput[i])
            return true;
    }
    return false;
}

// TODO [DS]: rewrite after new shape infer will be added
std::vector<VectorDims> MKLDNNReshapeNode::shapeInfer() const {
    const auto &memPtr = getParentEdgesAtPort(1)[0]->getMemory();

    const int32_t *sndInput = reinterpret_cast<const int32_t *>(memPtr.GetPtr());
    if (lastSecondInputValues.empty())
        lastSecondInputValues.resize(memPtr.getStaticDims()[0]);
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        lastSecondInputValues[i] = sndInput[i];
    }

    ngraph::OutputVector inputsForShapeInfer;
    inputsForShapeInfer.push_back(std::make_shared<ngraph::opset1::Parameter>(opToShapeInfer->get_input_element_type(0),
                                                                              getParentEdgesAtPort(0)[0]->getMemory().GetShape().toPartialShape()));
    inputsForShapeInfer.push_back(std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i32, memPtr.getStaticDims(), lastSecondInputValues));
    const auto localShapeInferOp = opToShapeInfer->clone_with_new_inputs(inputsForShapeInfer);

    localShapeInferOp->validate_and_infer_types();

    std::vector<VectorDims> newOutputShapes(outputShapes.size());
    for (size_t i = 0; i < newOutputShapes.size(); i++) {
        const auto &partShape = localShapeInferOp->get_output_partial_shape(i);
        if (partShape.is_dynamic())
            IE_THROW(NotImplemented) << "CPU plug-in doesn't support default shape infer for nodes with internal dynamism";
        newOutputShapes[i] = partShape.get_shape();
    }
    return newOutputShapes;
}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision inPrec = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrec = getOriginalOutputPrecisionAtPort(0);
    InferenceEngine::Precision secondInPrc = InferenceEngine::Precision::I32;

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = false;
        config.inConfs[i].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i > 0 ? secondInPrc : inPrec), getInputShapeAtPort(i));
    }
    config.outConfs.resize(1);
    // TODO [DS]: inplace
    config.outConfs[0].inPlace = isDynamicNode() ? -1 : 0;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
}

void MKLDNNReshapeNode::executeDynamicImpl(mkldnn::stream strm) {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const auto count = srcMemPtr->GetShape().getElementsCount();
    if (count != dstMemPtr->GetShape().getElementsCount())
        IE_THROW() << errorPrefix << " has different elements number in input and output buffers";
    cpu_memcpy(dstMemPtr->GetPtr(), srcMemPtr->GetPtr(), count * MKLDNNExtensionUtils::sizeOfDataType(srcMemPtr->GetDataType()));
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape;
}
REG_MKLDNN_PRIM_FOR(MKLDNNReshapeNode, Reshape);
