// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shapeof.h"
#include "openvino/opsets/opset1.hpp"
#include "shape_inference/custom/shapeof.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool ShapeOf::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v0::ShapeOf::get_type_info_static(),
                    ov::op::v3::ShapeOf::get_type_info_static())) {
            errorMessage = "Node is not an instance of ShapeOf form the operation set v1 or v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ShapeOf::ShapeOf(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, ShapeOfShapeInferFactory()) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ShapeOf layer with name '" + getName() + "' ";
        if (op->get_input_partial_shape(0).size() == 0)
            OPENVINO_THROW(errorPrefix, "gets unsupported input 0D tensor (scalar)");
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void ShapeOf::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        OPENVINO_THROW(errorPrefix, "has incorrect number of input edges: ", getParentEdges().size());
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, "has incorrect number of output edges: ", getChildEdges().size());
}

void ShapeOf::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, precision}},
                        {{LayoutType::ncsp, ov::element::i32}},
                        impl_desc_type::ref);
}

void ShapeOf::initOptimalPrimitiveDescriptor() {
    // Mimic the parent node memory desc to avoid extra reorder
    auto parentEdge = getParentEdgeAt(0);
    auto parent = parentEdge->getParent();
    auto parentPd = parent->getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(parentPd,
        parent->getTypeStr(), " ",
        parent->getName(),
        "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& parentConfig = parentPd->getConfig();
    auto mem_desc = parentConfig.outConfs[parentEdge->getInputNum()].getMemDesc();

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "ShapeOf ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selected_pd->getConfig();
    config.inConfs.front().setMemDesc(mem_desc);
    //bypass any checks, we enforce the parent descriptor
    selected_pd->setConfig(config);
}

bool ShapeOf::isExecutable() const {
    return true;
}

void ShapeOf::execute(dnnl::stream strm) {
    auto inPtr = getSrcMemoryAtPort(0);
    auto outPtr = getDstMemoryAtPort(0);
    auto&& inDims = inPtr->getStaticDims();
    size_t dimsCount = inDims.size();
    if (outPtr->getStaticDims().size() != 1 || dimsCount != outPtr->getStaticDims()[0])
        OPENVINO_THROW(errorPrefix, "has inconsistent input shape and output size");

    auto* dst = outPtr->getDataAs<int>();

    for (size_t i = 0; i < dimsCount; i++) {
        dst[i] = inDims[i];
    }
}

bool ShapeOf::created() const {
    return getType() == Type::ShapeOf;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
