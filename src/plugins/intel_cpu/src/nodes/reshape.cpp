// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.h"
#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <openvino/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include <utils/shape_inference/shape_inference.hpp>

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

Reshape::Reshape(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
        Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";

    if (isDynamicNode()) {
        auto checkSecondInput = [](const std::shared_ptr<ngraph::Node>& op, const std::string opType) {
            if (op->get_input_partial_shape(1).is_dynamic()) {
                IE_THROW() << "CPU plug-in doesn't support " << opType << " node with non static second input";
            }
        };

        if (std::dynamic_pointer_cast<const ov::opset1::Reshape>(op)) {
            checkSecondInput(op, "Reshape");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Squeeze>(op)) {
            if (op->get_input_size() == 1)
                IE_THROW() << "CPU plug-in doesn't support Squeeze node with inputs num equal 1";
            checkSecondInput(op, "Squeeze");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Unsqueeze>(op)) {
            checkSecondInput(op, "Unsqueeze");
        } else {
            IE_THROW() << "Unsupported operation type via reshape node";
        }
    }
}

bool Reshape::needShapeInfer() const {
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

std::vector<VectorDims> Reshape::shapeInfer() const {
    const auto &memPtr = getParentEdgesAtPort(1)[0]->getMemory();

    const int32_t *sndInput = reinterpret_cast<const int32_t *>(memPtr.GetPtr());
    if (lastSecondInputValues.empty())
        lastSecondInputValues.resize(memPtr.getStaticDims()[0]);
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        lastSecondInputValues[i] = sndInput[i];
    }

    return shapeInferGeneric(PortMask(1));
}

void Reshape::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
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

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace(-1);
        config.inConfs[i].constant(false);
        config.inConfs[i].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i > 0 ? secondInPrc : inPrec), getInputShapeAtPort(i)));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(0);
    config.outConfs[0].constant(false);
    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void Reshape::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Reshape::created() const {
    return getType() == Type::Reshape;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
