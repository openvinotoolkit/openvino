// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include <cstring>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
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

    const auto block_size = key_cache.size(2);
    CPU_NODE_ASSERT(block_size > 0, "expects non-zero key cache block size");
    CPU_NODE_ASSERT(value_cache.size(2) == block_size, "expects key/value cache to have identical block size");
    CPU_NODE_ASSERT(key_cache.size(1) == value_cache.size(1), "expects key/value cache to have identical kv head count");

    const auto kv_heads = key_cache.size(1);
    const auto bytes_per_key_token = key_cache.stride_bytes(2);
    const auto bytes_per_value_token = value_cache.stride_bytes(2);
    CPU_NODE_ASSERT(bytes_per_key_token > 0 && bytes_per_value_token > 0,
                    "expects non-zero token stride for key/value cache");

    CPU_NODE_ASSERT(block_indices_begins.size(0) == block_update_indices_begins.size(0),
                    "expects block_indices_begins and block_update_indices_begins to have same length");

    const auto seq_count = block_update_indices_begins.size(0) == 0 ? 0 : block_update_indices_begins.size(0) - 1;
    for (size_t seq = 0; seq < seq_count; seq++) {
        const auto block_begin = static_cast<size_t>(block_indices_begins.ptr<int32_t>()[seq]);
        const auto block_end = static_cast<size_t>(block_indices_begins.ptr<int32_t>()[seq + 1]);

        const auto upd_begin = static_cast<size_t>(block_update_indices_begins.ptr<int32_t>()[seq]);
        const auto upd_end = static_cast<size_t>(block_update_indices_begins.ptr<int32_t>()[seq + 1]);

        CPU_NODE_ASSERT(block_begin <= block_end && block_end <= block_indices.size(0),
                        "invalid block_indices_begins range for sequence ",
                        seq);
        CPU_NODE_ASSERT(upd_begin <= upd_end && upd_end * 2 <= block_update_indices.size(0),
                        "invalid block_update_indices_begins range for sequence ",
                        seq);

        const auto block_count = block_end - block_begin;
        for (size_t upd = upd_begin; upd < upd_end; upd++) {
            const auto src_logical = block_update_indices.ptr<int32_t>()[2 * upd];
            const auto dst_logical = block_update_indices.ptr<int32_t>()[2 * upd + 1];

            CPU_NODE_ASSERT(src_logical >= 0 && dst_logical >= 0,
                            "expects non-negative block update indices");

            const auto src_logical_u = static_cast<size_t>(src_logical);
            const auto dst_logical_u = static_cast<size_t>(dst_logical);

            const auto src_block_local = src_logical_u / block_size;
            const auto dst_block_local = dst_logical_u / block_size;
            const auto src_token = src_logical_u % block_size;
            const auto dst_token = dst_logical_u % block_size;

            CPU_NODE_ASSERT(src_block_local < block_count && dst_block_local < block_count,
                            "block update index is out of range for sequence ",
                            seq);

            const auto src_block = static_cast<size_t>(block_indices.ptr<int32_t>()[block_begin + src_block_local]);
            const auto dst_block = static_cast<size_t>(block_indices.ptr<int32_t>()[block_begin + dst_block_local]);

            CPU_NODE_ASSERT(src_block < key_cache.size(0) && dst_block < key_cache.size(0),
                            "key cache block index out of range");
            CPU_NODE_ASSERT(src_block < value_cache.size(0) && dst_block < value_cache.size(0),
                            "value cache block index out of range");

            for (size_t h = 0; h < kv_heads; h++) {
                auto* key_src = key_cache.ptr_v(src_block, h, src_token, 0);
                auto* key_dst = key_cache.ptr_v(dst_block, h, dst_token, 0);
                std::memmove(key_dst, key_src, bytes_per_key_token);

                auto* value_src = value_cache.ptr_v(src_block, h, src_token, 0);
                auto* value_dst = value_cache.ptr_v(dst_block, h, dst_token, 0);
                std::memmove(value_dst, value_src, bytes_per_value_token);
            }
        }
    }

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
