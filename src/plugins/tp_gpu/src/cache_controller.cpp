// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_controller.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"

namespace ov {
namespace tp_gpu {

namespace {

/// Cache ports are named "key_cache.<layer>" / "value_cache.<layer>".
bool is_cache_port(const ov::Output<const ov::Node>& port) {
    for (const auto& name : port.get_names()) {
        if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0)
            return true;
    }
    return false;
}

bool is_value_cache_port(const ov::Output<const ov::Node>& port) {
    for (const auto& name : port.get_names()) {
        if (name.rfind("value_cache.", 0) == 0)
            return true;
    }
    return false;
}

/// Tokens per block, as the GPU plugin lays the cache out. It is not written
/// anywhere in the model; the sparse-attention layout is the one that can be
/// told apart, by the block dimension of the value cache.
size_t block_size_of(const ov::Shape& value_cache_shape) {
    constexpr size_t kSparseAttentionBlockSize = 256;
    constexpr size_t kBlockSize = 16;
    return value_cache_shape[2] == kSparseAttentionBlockSize ? kSparseAttentionBlockSize : kBlockSize;
}

/// Cache tensors live in device memory, and a device-to-device copy only works
/// through the owning plugin's own tensor type.  Wrapping one in a generic
/// region-of-interest tensor produces something the plugin refuses to
/// recognize, so regions are expressed as byte offsets into the tensor instead.
ov::IRemoteTensor& as_remote(const ov::SoPtr<ov::ITensor>& tensor) {
    auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    OPENVINO_ASSERT(remote, "[TP_GPU] The paged-attention cache is expected to live in device memory");
    return *remote;
}

}  // namespace

std::shared_ptr<CacheController> CacheController::create(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled) {
    auto controller = std::shared_ptr<CacheController>(new CacheController());

    for (const auto& compiled : rank_compiled) {
        Rank rank;
        for (const auto& port : compiled->inputs()) {
            if (!is_cache_port(port))
                continue;

            const auto& pshape = port.get_partial_shape();
            OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() == 4,
                            "[TP_GPU] Cache port '", port.get_any_name(), "' has shape ", pshape,
                            "; a paged-attention cache is [blocks, kv_heads, ..., ...]");
            OPENVINO_ASSERT(port.get_element_type().is_static(),
                            "[TP_GPU] Cache port '", port.get_any_name(),
                            "' left its precision open; the compiled rank model is expected to have "
                            "settled it");

            // Dimension 0 is the block count, decided at allocation time; the
            // rest is fixed by the compiled model and describes one block.
            ov::Shape shape{0, 0, 0, 0};
            for (size_t d = 1; d < 4; ++d) {
                OPENVINO_ASSERT(pshape[d].is_static(),
                                "[TP_GPU] Cache port '", port.get_any_name(), "' has shape ", pshape,
                                "; everything but the block count has to be known to size a block");
                shape[d] = static_cast<size_t>(pshape[d].get_length());
            }

            rank.ports.push_back(port);
            rank.precisions.push_back(port.get_element_type());
            rank.shapes.push_back(shape);
        }

        if (rank.ports.empty()) {
            // Not a paged-attention model: nothing for the plugin to own.
            OPENVINO_ASSERT(controller->m_ranks.empty(),
                            "[TP_GPU] Rank models disagree on whether they use a paged-attention cache");
            return nullptr;
        }

        rank.context = compiled->get_context();
        rank.tensors.resize(rank.ports.size());
        controller->m_ranks.push_back(std::move(rank));
    }

    OPENVINO_ASSERT(!controller->m_ranks.empty(), "[TP_GPU] Cache controller built over no ranks");

    // A block spans every layer and every rank -- that total is what a
    // scheduler budgets its memory against.
    size_t value_ports = 0;
    ov::Shape first_value_shape;
    for (const auto& rank : controller->m_ranks) {
        OPENVINO_ASSERT(rank.ports.size() == controller->m_ranks.front().ports.size(),
                        "[TP_GPU] Ranks disagree on the number of cache ports");
        for (size_t i = 0; i < rank.ports.size(); ++i) {
            const auto& shape = rank.shapes[i];
            controller->m_block_size_in_bytes +=
                shape[1] * shape[2] * shape[3] * rank.precisions[i].size();
            if (is_value_cache_port(rank.ports[i])) {
                if (value_ports == 0)
                    first_value_shape = shape;
                ++value_ports;
            }
        }
    }

    OPENVINO_ASSERT(value_ports > 0, "[TP_GPU] Paged-attention model without a value cache port");
    controller->m_num_layers = value_ports / controller->m_ranks.size();
    controller->m_block_size = block_size_of(first_value_shape);

    return controller;
}

size_t CacheController::get_num_cache_tensors() const {
    size_t tensors = 0;
    for (const auto& rank : m_ranks)
        tensors += rank.ports.size();
    return tensors;
}

void CacheController::allocate_cache_if_needed(size_t num_blocks) {
    if (num_blocks <= m_allocated_blocks)
        return;

    for (auto& rank : m_ranks) {
        for (size_t i = 0; i < rank.ports.size(); ++i) {
            auto shape = rank.shapes[i];
            shape[0] = num_blocks;

            auto grown = rank.context->create_tensor(rank.precisions[i], shape, {});

            // Growing keeps what the cache already holds: the scheduler is
            // still tracking those blocks and will read them back.  Only the
            // block count grows, so the old content is a prefix of the new
            // tensor and copies straight to offset zero.
            if (const auto& previous = rank.tensors[i]) {
                as_remote(previous).copy_to(grown._ptr,
                                            /*src_offset=*/0,
                                            /*dst_offset=*/0,
                                            previous->get_shape());
            }

            rank.tensors[i] = ov::SoPtr<ov::ITensor>(grown._ptr, grown._so);
        }
    }

    m_allocated_blocks = num_blocks;
    ++m_generation;
}

void CacheController::copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
    if (block_copy_map.empty())
        return;

    for (auto& rank : m_ranks) {
        for (size_t i = 0; i < rank.ports.size(); ++i) {
            const auto& tensor = rank.tensors[i];
            OPENVINO_ASSERT(tensor, "[TP_GPU] copy_blocks before the cache was allocated");

            auto block_shape = tensor->get_shape();
            const size_t blocks = block_shape[0];
            block_shape[0] = 1;

            // Blocks are the outermost dimension, so one block is a
            // contiguous run and its offset is just its index times the
            // outermost stride.
            const size_t block_bytes = tensor->get_strides().front();
            auto& remote = as_remote(tensor);

            for (const auto& [source, destinations] : block_copy_map) {
                OPENVINO_ASSERT(source < blocks, "[TP_GPU] copy_blocks source block ", source,
                                " is outside the ", blocks, " allocated blocks");
                for (size_t destination : destinations) {
                    OPENVINO_ASSERT(destination < blocks, "[TP_GPU] copy_blocks destination block ",
                                    destination, " is outside the ", blocks, " allocated blocks");
                    remote.copy_to(tensor._ptr,
                                   source * block_bytes,
                                   destination * block_bytes,
                                   block_shape);
                }
            }
        }
    }
}

void CacheController::zero_blocks(const std::set<size_t>&) {
    // A KV cache block is written in full before anything reads it, so a
    // recycled block never contributes stale values.
}

void CacheController::clear() {
    for (auto& rank : m_ranks)
        std::fill(rank.tensors.begin(), rank.tensors.end(), ov::SoPtr<ov::ITensor>{});
    m_allocated_blocks = 0;
    ++m_generation;
}

}  // namespace tp_gpu
}  // namespace ov
