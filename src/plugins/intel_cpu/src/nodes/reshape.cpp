// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.h"

#include "common/cpu_memcpy.h"
#include <openvino/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Reshape::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::opset1::Reshape::get_type_info_static(),
                    ov::opset1::Squeeze::get_type_info_static(),
                    ov::opset1::Unsqueeze::get_type_info_static())) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reshape::Reshape(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context) :
        Node(op, context, NgraphShapeInferFactory(op, PortMask(1))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (isDynamicNode()) {
        auto checkSecondInput = [](const std::shared_ptr<ov::Node>& op, const std::string &opType) {
            if (op->get_input_partial_shape(1).is_dynamic()) {
                IE_THROW() << "CPU plug-in doesn't support " << opType << " node with non static second input";
            }
        };

        if (op->get_type_info() == ov::opset1::Reshape::get_type_info_static()) {
            checkSecondInput(op, getTypeStr());
        } else if (op->get_type_info() == ov::opset1::Squeeze::get_type_info_static()) {
            if (op->get_input_size() == 1)
                IE_THROW() << "CPU plug-in doesn't support Squeeze node with inputs num equal 1";
            checkSecondInput(op, getTypeStr());
        } else if (op->get_type_info() == ov::opset1::Unsqueeze::get_type_info_static()) {
            checkSecondInput(op, getTypeStr());
        } else {
            IE_THROW() << "Unsupported operation type via reshape node";
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

    const auto shapePrc = mem.getDesc().getPrecision();
    if (shapePrc == Precision::I64) {
        const auto sndInput = reinterpret_cast<const int64_t *>(mem.GetPtr());
        for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
            if (lastSecondInputValues[i] != sndInput[i]) {
                for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
                    lastSecondInputValues[i] = sndInput[i];
                }
                return true;
            }
        }
    } else if (shapePrc == Precision::I32) {
        const auto sndInput = reinterpret_cast<const int32_t *>(mem.GetPtr());
        for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
            if (lastSecondInputValues[i] != sndInput[i]) {
                for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
                    lastSecondInputValues[i] = sndInput[i];
                }
                return true;
            }
        }
    }

    return false;
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

    auto inPrec = getOriginalInputPrecisionAtPort(0);
    Precision secondInPrc = Precision::I32;
    if (getOriginalInputPrecisions().size() > 1) {
        secondInPrc = getOriginalInputPrecisionAtPort(1);
    }
    const auto &outPrec = getOriginalOutputPrecisionAtPort(0);

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    bool canBeInPlace = true;

    // CVS-81059 : disable inPlace in following case since it won't be satisfied by framework
    if (!isConstant() && getParentEdgeAt(0)->getParent()->isConstant())
        canBeInPlace = false;

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace(-1);
        config.inConfs[i].constant(false);
        config.inConfs[i].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i == 0 ? inPrec : secondInPrc), getInputShapeAtPort(i)));
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
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();

    auto srcPtr = static_cast<uint8_t*>(srcMemPtr->GetPtr());
    auto dstPtr = static_cast<uint8_t*>(dstMemPtr->GetPtr());

    if (dstPtr != srcPtr) {
        cpu_memcpy(dstPtr, srcPtr, dstMemPtr->GetSize());
    }
}

bool Reshape::isExecutable() const {
    bool inPlaceEnabled =
        getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].inPlace() >= 0;
    return !inPlaceEnabled;
}

bool Reshape::created() const {
    return getType() == Type::Reshape;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
