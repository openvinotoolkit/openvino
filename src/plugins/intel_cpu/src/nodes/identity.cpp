// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity.hpp"

#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/cpu_memcpy.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/identity.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

bool Identity::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v16::Identity::get_type_info_static()) {
            errorMessage = "Only Identity operation from the opset16 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Identity::Identity(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void Identity::getSupportedDescriptors() {
    if (getParentEdges().size() != 1) {
        CPU_NODE_THROW("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("has incorrect number of output edges.");
    }
}

void Identity::initSupportedPrimitiveDescriptors() {
    auto in_prc = getOriginalInputPrecisionAtPort(0);
    auto out_prc = getOriginalOutputPrecisionAtPort(0);

    if (in_prc != out_prc) {
        CPU_NODE_THROW("has to have the same dtype for input and output nodes. src: ", in_prc, ", dst: ", out_prc);
    }

    m_out_prc = out_prc;

    addSupportedPrimDesc({{LayoutType::ncsp, in_prc}}, {{LayoutType::ncsp, out_prc}}, ref_any);
}

void Identity::prepareParams() {
    VectorDims out_shape = getDstMemoryAtPort(0)->getShape().getStaticDims();
    m_element_num = std::accumulate(out_shape.begin(), out_shape.end(), 1UL, std::multiplies<>());
}

bool Identity::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

bool Identity::created() const {
    return getType() == Type::Identity;
}

bool Identity::canBeInPlace() const {
    return getSrcMemoryAtPort(0) == getDstMemoryAtPort(0);
}

void Identity::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (!canBeInPlace()) {
        auto* input = getSrcDataAtPort(0);
        auto* output = getDstDataAtPort(0);

        cpu_parallel_memcpy(output, input, m_out_prc.size() * m_element_num);
    }
}

void Identity::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
