// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/cpu_memcpy.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shape_inference/custom/reshape.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool Reshape::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const ov::op::v1::Reshape>(op) && !ov::as_type_ptr<const ov::op::v0::Squeeze>(op) &&
            !ov::as_type_ptr<const ov::op::v0::Unsqueeze>(op)) {
            errorMessage = "Only v1 Reshape, v0 Squeeze and v0 Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reshape::Reshape(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, ReshapeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (isDynamicNode()) {
        auto checkSecondInput = [this](const std::shared_ptr<ov::Node>& op,
                                       [[maybe_unused]] const std::string& opType) {
            if (op->get_input_partial_shape(1).is_dynamic()) {
                CPU_NODE_THROW("has non static second input");
            }
        };

        if (ov::as_type_ptr<const ov::op::v1::Reshape>(op)) {
            checkSecondInput(op, "Reshape");
        } else if (ov::as_type_ptr<const ov::op::v0::Squeeze>(op)) {
            if (op->get_input_size() == 1) {
                CPU_NODE_THROW("has inputs num equal 1");
            }
            checkSecondInput(op, "Squeeze");
        } else if (ov::as_type_ptr<const ov::op::v0::Unsqueeze>(op)) {
            checkSecondInput(op, "Unsqueeze");
        } else {
            CPU_NODE_THROW("Unsupported operation type via reshape node");
        }
    }
}

bool Reshape::needShapeInfer() const {
    const auto& mem = getParentEdgeAt(1)->getMemory();
    if (lastSecondInputValues.empty()) {
        lastSecondInputValues.resize(mem.getStaticDims()[0], 0);
    }
    const auto* sndInput = mem.getDataAs<const int32_t>();
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        if (lastSecondInputValues[i] != sndInput[i]) {
            for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
                lastSecondInputValues[i] = sndInput[i];
            }
            return true;
        }
    }
    return inputShapesModified();
}

void Reshape::getSupportedDescriptors() {
    if (none_of(getParentEdges().size(), 1U, 2U)) {
        CPU_NODE_THROW("Incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("Incorrect number of output edges");
    }
}

void Reshape::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inPrec = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outPrec = getOriginalOutputPrecisionAtPort(0);
    ov::element::Type secondInPrc = ov::element::i32;

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec) {
        inPrec = outPrec;
    }

    bool canBeInPlace = true;

    // CVS-81059 : disable inPlace in following case since it won't be satisfied by framework
    if (!isConstant() && getParentEdgeAt(0)->getParent()->isConstant()) {
        canBeInPlace = false;
    }

    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace(0 == i && canBeInPlace ? 0 : -1);
        config.inConfs[i].constant(false);
        config.inConfs[i].setMemDesc(
            creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i > 0 ? secondInPrc : inPrec), getInputShapeAtPort(i)));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(canBeInPlace ? 0 : -1);
    config.outConfs[0].constant(false);
    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void Reshape::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Reshape::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto srcMemPtr = getSrcMemoryAtPort(0);
    auto dstMemPtr = getDstMemoryAtPort(0);

    auto* srcPtr = static_cast<uint8_t*>(srcMemPtr->getData());
    auto* dstPtr = static_cast<uint8_t*>(dstMemPtr->getData());

    if (dstPtr != srcPtr) {
        cpu_memcpy(dstPtr, srcPtr, dstMemPtr->getSize());
    }
}

bool Reshape::neverExecute() const {
    bool inPlaceEnabled = false;
    if (const auto* prim_desc = getSelectedPrimitiveDescriptor()) {
        const auto& config = prim_desc->getConfig();
        if (config.inConfs[0].inPlace() >= 0 || config.outConfs[0].inPlace() >= 0) {
            inPlaceEnabled = true;
        }
    }
    return inPlaceEnabled;
}

bool Reshape::isExecutable() const {
    return !neverExecute();
}

bool Reshape::created() const {
    return getType() == Type::Reshape;
}

}  // namespace ov::intel_cpu::node
