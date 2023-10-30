// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.h"
#include "utils.hpp"
#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <openvino/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include "shape_inference/custom/reshape.hpp"

#include "common/cpu_memcpy.h"

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Reshape::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::opset1::Reshape>(op) &&
            !std::dynamic_pointer_cast<const ov::opset1::Squeeze>(op) &&
                !std::dynamic_pointer_cast<const ov::opset1::Unsqueeze>(op)) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reshape::Reshape(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, ReshapeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";

    if (isDynamicNode()) {
        auto checkSecondInput = [](const std::shared_ptr<ngraph::Node>& op, const std::string opType) {
            if (op->get_input_partial_shape(1).is_dynamic()) {
                OPENVINO_THROW("CPU plug-in doesn't support ", opType, " node with non static second input");
            }
        };

        if (std::dynamic_pointer_cast<const ov::opset1::Reshape>(op)) {
            checkSecondInput(op, "Reshape");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Squeeze>(op)) {
            if (op->get_input_size() == 1)
                OPENVINO_THROW("CPU plug-in doesn't support Squeeze node with inputs num equal 1");
            checkSecondInput(op, "Squeeze");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Unsqueeze>(op)) {
            checkSecondInput(op, "Unsqueeze");
        } else {
            OPENVINO_THROW("Unsupported operation type via reshape node");
        }
    }
}

bool Reshape::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    const auto& mem = getParentEdgesAtPort(1)[0]->getMemory();
    if (lastSecondInputValues.empty()) {
        lastSecondInputValues.resize(mem.getStaticDims()[0], 0);
    }
    const int32_t *sndInput = reinterpret_cast<const int32_t *>(mem.getData());
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        if (lastSecondInputValues[i] != sndInput[i]) {
            for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
                lastSecondInputValues[i] = sndInput[i];
            }
            return true;
        }
    }
    return false;
}

void Reshape::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        OPENVINO_THROW("Incorrect number of input edges for layer ", getName());
    if (getChildEdges().empty())
        OPENVINO_THROW("Incorrect number of output edges for layer ", getName());
}

void Reshape::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision inPrec = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrec = getOriginalOutputPrecisionAtPort(0);
    InferenceEngine::Precision secondInPrc = InferenceEngine::Precision::I32;

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    bool canBeInPlace = true;

    // CVS-81059 : disable inPlace in following case since it won't be satisfied by framework
    if (!isConstant() && getParentEdgeAt(0)->getParent()->isConstant())
        canBeInPlace = false;

    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace(0 == i && canBeInPlace ? 0 : -1);
        config.inConfs[i].constant(false);
        config.inConfs[i].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i > 0 ? secondInPrc : inPrec), getInputShapeAtPort(i)));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(canBeInPlace ? 0 : -1);
    config.outConfs[0].constant(false);
    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void Reshape::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reshape::execute(dnnl::stream strm) {
    auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();

    auto srcPtr = static_cast<uint8_t*>(srcMemPtr->getData());
    auto dstPtr = static_cast<uint8_t*>(dstMemPtr->getData());

    if (dstPtr != srcPtr) {
        cpu_memcpy(dstPtr, srcPtr, dstMemPtr->getSize());
    }
}

bool Reshape::isExecutable() const {
    bool inPlaceEnabled = false;
    if (auto prim_desc = getSelectedPrimitiveDescriptor()) {
        auto& config = prim_desc->getConfig();
        if (config.inConfs[0].inPlace() >= 0 ||
            config.outConfs[0].inPlace() >= 0) {
            inPlaceEnabled = true;
        }
    }
    return !inPlaceEnabled;
}

bool Reshape::created() const {
    return getType() == Type::Reshape;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
