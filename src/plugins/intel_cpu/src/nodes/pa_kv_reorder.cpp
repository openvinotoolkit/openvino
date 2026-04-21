// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>

#include "config.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/cache_reorder.hpp"
#include "nodes/paged_attn.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

bool PaKVReorder::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_name() != std::string("PaKVReorder")) {
            errorMessage = "Unsupported operation type for PaKVReorder CPU node: " + std::string(op->get_type_name());
            return false;
        }

        if (op->get_input_size() != 6) {
            errorMessage = "PaKVReorder expects 6 inputs.";
            return false;
        }

        if (op->get_output_size() != 1) {
            errorMessage = "PaKVReorder expects 1 output.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

PaKVReorder::PaKVReorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void PaKVReorder::getSupportedDescriptors() {
    if (getParentEdges().size() != 6) {
        CPU_NODE_THROW("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("has incorrect number of output edges.");
    }
}

void PaKVReorder::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::dynamic},
                          {LayoutType::ncsp, ov::element::dynamic},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, ov::element::u8}},
                         impl_desc_type::ref_any);
}

void PaKVReorder::createPrimitive() {
    // Determine quantization mode once at primitive creation time
    const auto& cpuConfig = context->getConfig();
    auto keyCachePrecision = getOriginalInputPrecisionAtPort(0);
    auto valueCachePrecision = getOriginalInputPrecisionAtPort(1);

    m_key_by_channel = PagedAttention::isQuantByChannel(cpuConfig.keyCacheQuantMode, keyCachePrecision, true);
    m_value_by_channel = PagedAttention::isQuantByChannel(cpuConfig.valueCacheQuantMode, valueCachePrecision, false);
}

void PaKVReorder::execute([[maybe_unused]] const dnnl::stream& strm) {
    PlainTensor key_cache(getSrcMemoryAtPort(0));
    PlainTensor value_cache(getSrcMemoryAtPort(1));
    PlainTensor block_indices(getSrcMemoryAtPort(2));
    PlainTensor block_indices_begins(getSrcMemoryAtPort(3));
    PlainTensor block_update_indices(getSrcMemoryAtPort(4));
    PlainTensor block_update_indices_begins(getSrcMemoryAtPort(5));

    block_indices.assert_dims({0}, true);
    block_indices_begins.assert_dims({0}, true);
    block_update_indices.assert_dims({0}, true);
    block_update_indices_begins.assert_dims({0}, true);

    CPU_NODE_ASSERT(key_cache.m_rank == 4, "expects 4D key cache tensor");
    CPU_NODE_ASSERT(value_cache.m_rank == 4, "expects 4D value cache tensor");

    CPU_NODE_ASSERT(key_cache.size(1) == value_cache.size(1),
                    "expects key/value cache to have identical kv head count");

    CPU_NODE_ASSERT(block_indices_begins.size(0) == block_update_indices_begins.size(0),
                    "expects block_indices_begins and block_update_indices_begins to have same length");

    // Delegate to optimized kernel implementation with parallel execution
    // Quantization configuration (m_key_by_channel, m_value_by_channel) was determined at createPrimitive time
    // Thread-local buffers are used internally for quantization scratch space
    ov::Extensions::Cpu::XARCH::reorder_kv_cache(key_cache,
                                                 value_cache,
                                                 block_indices,
                                                 block_indices_begins,
                                                 block_update_indices,
                                                 block_update_indices_begins,
                                                 m_key_by_channel,
                                                 m_value_by_channel,
                                                 context->getCpuParallel());

    if (getChildEdges().empty()) {
        return;
    }

    auto* out = getDstDataAtPort(0);
    if (out != nullptr && !getDstMemoryAtPort(0)->getShape().hasZeroDims()) {
        std::memset(out, 0, getDstMemoryAtPort(0)->getDesc().getCurrentMemSize());
    }
}

void PaKVReorder::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool PaKVReorder::created() const {
    return getType() == Type::PaKVReorder;
}

}  // namespace ov::intel_cpu::node
