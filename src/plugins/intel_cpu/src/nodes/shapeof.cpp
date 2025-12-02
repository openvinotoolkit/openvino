// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shapeof.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/shape_of.hpp"
#include "shape_inference/custom/shapeof.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool ShapeOf::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (none_of(op->get_type_info(),
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

ShapeOf::ShapeOf(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, ShapeOfShapeInferFactory()) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        CPU_NODE_ASSERT(op->get_input_partial_shape(0).size() != 0, "gets unsupported input 0D tensor (scalar)");
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void ShapeOf::getSupportedDescriptors() {
    CPU_NODE_ASSERT(getParentEdges().size() == 1, "has incorrect number of input edges: ", getParentEdges().size());
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges: ", getChildEdges().size());
}

void ShapeOf::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, precision}}, {{LayoutType::ncsp, ov::element::i32}}, impl_desc_type::ref);
}

void ShapeOf::initOptimalPrimitiveDescriptor() {
    // Mimic the parent node memory desc to avoid extra reorder
    auto parentEdge = getParentEdgeAt(0);
    auto parent = parentEdge->getParent();
    auto* parentPd = parent->getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(parentPd,
                    "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& parentConfig = parentPd->getConfig();
    auto mem_desc = parentConfig.outConfs[parentEdge->getInputNum()].getMemDesc();

    auto* selected_pd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selected_pd,
                    "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selected_pd->getConfig();
    config.inConfs.front().setMemDesc(mem_desc);
    // bypass any checks, we enforce the parent descriptor
    selected_pd->setConfig(config);
}

void ShapeOf::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto inPtr = getSrcMemoryAtPort(0);
    auto outPtr = getDstMemoryAtPort(0);
    auto&& inDims = inPtr->getStaticDims();
    size_t dimsCount = inDims.size();
    CPU_NODE_ASSERT(outPtr->getStaticDims().size() == 1 && dimsCount == outPtr->getStaticDims()[0],
                    "has inconsistent input shape and output size");

    auto* dst = outPtr->getDataAs<int>();

    for (size_t i = 0; i < dimsCount; i++) {
        dst[i] = inDims[i];
    }
}

bool ShapeOf::created() const {
    return getType() == Type::ShapeOf;
}

}  // namespace ov::intel_cpu::node
