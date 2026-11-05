// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_controller.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"

namespace ov {
namespace tp {

namespace {

/// Cache ports are named "key_cache.<layer>" / "value_cache.<layer>"; returns that name, or empty.
std::string cache_name(const ov::Output<const ov::Node>& port) {
    for (const auto& name : port.get_names()) {
        if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0)
            return name;
    }
    return {};
}

size_t layer_of(const std::string& name) {
    return static_cast<size_t>(std::stoul(name.substr(name.find('.') + 1)));
}

/// Tokens per block, as the GPU plugin lays the cache out. It is not written anywhere in the model; the
/// sparse-attention layout is the one that can be told apart, by the block dimension of the value cache.
size_t block_size_of(const ov::Shape& value_cache_shape) {
    constexpr size_t kSparseAttentionBlockSize = 256;
    constexpr size_t kBlockSize = 16;
    return value_cache_shape[2] == kSparseAttentionBlockSize ? kSparseAttentionBlockSize : kBlockSize;
}

size_t bytes_of(const ov::element::Type& type, size_t elements) {
    return (elements * type.bitwidth() + 7) / 8;
}

size_t block_elements(const ov::Shape& shape) {
    return shape[1] * shape[2] * shape[3];
}

/// Cache tensors live in device memory, and a device-to-device copy only works through the owning plugin's own
/// tensor type, so regions are expressed as byte offsets into the tensor.
ov::IRemoteTensor& as_remote(const ov::SoPtr<ov::ITensor>& tensor) {
    auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    OPENVINO_ASSERT(remote, "[TP] The paged-attention cache is expected to live in device memory");
    return *remote;
}

ov::Shape with_blocks(ov::Shape shape, size_t blocks) {
    shape[0] = blocks;
    return shape;
}

}  // namespace

std::shared_ptr<CacheController> CacheController::create(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled,
                                                         const std::vector<std::string>& device_names,
                                                         std::shared_ptr<const ov::ICore> core,
                                                         size_t kv_replicas) {
    auto controller = std::shared_ptr<CacheController>(new CacheController());
    controller->m_core = std::move(core);

    for (size_t r = 0; r < rank_compiled.size(); ++r) {
        Rank rank;
        rank.device = device_names.at(r);
        for (const auto& port : rank_compiled[r]->inputs()) {
            if (cache_name(port).empty())
                continue;
            const auto& pshape = port.get_partial_shape();
            OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() == 4,
                            "[TP] Cache port '", port.get_any_name(), "' has shape ", pshape,
                            "; a paged-attention cache is [blocks, kv_heads, ..., ...]");
            OPENVINO_ASSERT(port.get_element_type().is_static(),
                            "[TP] Cache port '", port.get_any_name(),
                            "' left its precision open; the compiled rank model is expected to have settled it");
            ov::Shape shape{0, 0, 0, 0};
            for (size_t d = 1; d < 4; ++d) {
                OPENVINO_ASSERT(pshape[d].is_static(), "[TP] Cache port '", port.get_any_name(), "' has shape ", pshape,
                                "; everything but the block count has to be known to size a block");
                shape[d] = static_cast<size_t>(pshape[d].get_length());
            }
            const auto bytes = bytes_of(port.get_element_type(), block_elements(shape));
            rank.ports.push_back(port);
            rank.precisions.push_back(port.get_element_type());
            rank.shapes.push_back(shape);
            rank.block_bytes += bytes;
            rank.max_tensor_block_bytes = std::max(rank.max_tensor_block_bytes, bytes);
        }
        if (rank.ports.empty()) {
            OPENVINO_ASSERT(controller->m_ranks.empty(), "[TP] Rank models disagree on whether they use a paged-attention cache");
            return nullptr;
        }
        rank.context = rank_compiled[r]->get_context();
        controller->m_ranks.push_back(std::move(rank));
    }

    // The group in the model-level representation: every rank's KV heads, each shared head counted once.
    const auto& first = controller->m_ranks.front();
    auto& group = controller->m_group;
    group.kind = ov::CacheKind::KEY_VALUE;
    for (size_t i = 0; i < first.ports.size(); ++i) {
        size_t heads = 0;
        for (const auto& rank : controller->m_ranks) {
            OPENVINO_ASSERT(rank.ports.size() == first.ports.size() && cache_name(rank.ports[i]) == cache_name(first.ports[i]) &&
                                rank.precisions[i] == first.precisions[i],
                            "[TP] Ranks disagree on the cache ports");
            heads += rank.shapes[i][1];
        }
        const auto name = cache_name(first.ports[i]);
        const auto& shape = first.shapes[i];
        ov::CacheTensorInfo tensor;
        tensor.name = name;
        tensor.layer = layer_of(name);
        tensor.element_type = first.precisions[i];
        tensor.shape = ov::PartialShape{ov::Dimension::dynamic(),
                                        static_cast<int64_t>(heads / kv_replicas),
                                        static_cast<int64_t>(shape[2]),
                                        static_cast<int64_t>(shape[3])};
        group.block_size_in_bytes += bytes_of(tensor.element_type, heads / kv_replicas * shape[2] * shape[3]);
        if (name.rfind("value_cache.", 0) == 0 && group.block_size == 0)
            group.block_size = block_size_of(shape);
        group.tensors.push_back(std::move(tensor));
    }
    OPENVINO_ASSERT(group.block_size > 0, "[TP] Paged-attention model without a value cache port");
    return controller;
}

void CacheController::check_group(size_t group) const {
    OPENVINO_ASSERT(group == 0, "[TP] The model has one plugin-managed cache group (KEY_VALUE), not group ", group);
}

std::vector<ov::CacheGroupInfo> CacheController::get_groups() const {
    return {m_group};
}

size_t CacheController::get_available_memory() const {
    // Unaccounted memory (kernels, pools) is kept out the way the application did on one device.
    constexpr double used_memory_threshold = 1.1;
    size_t blocks = std::numeric_limits<size_t>::max();
    for (const auto& rank : m_ranks) {
        const auto memory = m_core->get_property(rank.device, ov::tp::internal::device_memory.name(), {})
                                .as<ov::tp::internal::DeviceMemory>();
        const auto used = static_cast<uint64_t>(static_cast<double>(memory.used) * used_memory_threshold);
        const uint64_t free = memory.total > used ? memory.total - used : 0;
        blocks = std::min<size_t>(blocks, free / rank.block_bytes);
        blocks = std::min<size_t>(blocks, memory.max_allocation / rank.max_tensor_block_bytes);
    }
    return blocks * m_group.block_size_in_bytes;
}

size_t CacheController::get_num_allocated_blocks(size_t group) const {
    check_group(group);
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_num_blocks;
}

void CacheController::check_allocated() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_num_blocks > 0,
                    "[TP] The paged-attention cache has not been allocated. Allocate it through the compiled "
                    "model's ov::plugin_managed_cache before inferring.");
}

void CacheController::zero(const ov::SoPtr<ov::ITensor>& tensor, size_t first_block, size_t num_blocks) const {
    constexpr size_t kMaxChunkBytes = 64u << 20;
    const size_t block_bytes = tensor->get_strides().front();
    const size_t chunk = std::max<size_t>(1, std::min(num_blocks, kMaxChunkBytes / block_bytes));
    const auto zeros = ov::make_tensor(tensor->get_element_type(), with_blocks(tensor->get_shape(), chunk));
    std::memset(zeros->data(), 0, zeros->get_byte_size());
    auto& remote = as_remote(tensor);
    for (size_t block = first_block; block < first_block + num_blocks; block += chunk) {
        const size_t count = std::min(chunk, first_block + num_blocks - block);
        remote.copy_from(zeros, 0, block * block_bytes, with_blocks(tensor->get_shape(), count));
    }
}

CacheController::RankTensors CacheController::build(size_t num_blocks, size_t keep, bool zero_new) const {
    RankTensors tensors(m_ranks.size());
    for (size_t r = 0; r < m_ranks.size(); ++r) {
        const auto& rank = m_ranks[r];
        for (size_t i = 0; i < rank.ports.size(); ++i) {
            auto created = rank.context->create_tensor(rank.precisions[i], with_blocks(rank.shapes[i], num_blocks), {});
            ov::SoPtr<ov::ITensor> tensor(created._ptr, created._so);
            if (keep > 0) {
                // Blocks are the outermost dimension: the kept blocks are a prefix of both tensors.
                as_remote(m_tensors[r][i]).copy_to(tensor._ptr, 0, 0, with_blocks(rank.shapes[i], keep));
            }
            if (zero_new && num_blocks > keep)
                zero(tensor, keep, num_blocks - keep);
            tensors[r].push_back(std::move(tensor));
        }
    }
    return tensors;
}

CacheController::RankTensors CacheController::placeholders() const {
    RankTensors tensors(m_ranks.size());
    for (size_t r = 0; r < m_ranks.size(); ++r) {
        for (size_t i = 0; i < m_ranks[r].ports.size(); ++i)
            tensors[r].push_back(ov::make_tensor(m_ranks[r].precisions[i], m_ranks[r].shapes[i]));
    }
    return tensors;
}

void CacheController::bind(const std::vector<std::weak_ptr<ov::IAsyncInferRequest>>& request,
                           const RankTensors& tensors) const {
    for (size_t r = 0; r < m_ranks.size(); ++r) {
        const auto rank_request = request[r].lock();
        if (!rank_request)
            continue;
        for (size_t i = 0; i < m_ranks[r].ports.size(); ++i)
            rank_request->set_tensor(m_ranks[r].ports[i], tensors[r][i]);
    }
}

void CacheController::bind_all(const RankTensors& tensors) {
    for (const auto& [id, request] : m_requests)
        bind(request, tensors);
}

void CacheController::allocate(size_t group, size_t num_blocks, bool zero_initialize) {
    check_group(group);
    OPENVINO_ASSERT(num_blocks > 0, "[TP] A cache of 0 blocks cannot be allocated; deallocate() frees the cache");
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_num_blocks == 0, "[TP] The cache is already allocated (", m_num_blocks, " blocks); use reallocate()");
    auto tensors = build(num_blocks, 0, zero_initialize);
    try {
        bind_all(tensors);
    } catch (...) {
        bind_all(placeholders());
        throw;
    }
    m_tensors = std::move(tensors);
    m_num_blocks = num_blocks;
}

void CacheController::reallocate(size_t group, size_t num_blocks, bool zero_initialize_new_blocks) {
    check_group(group);
    OPENVINO_ASSERT(num_blocks > 0, "[TP] A cache of 0 blocks cannot be allocated; deallocate() frees the cache");
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_num_blocks > 0, "[TP] The cache is not allocated; use allocate()");
    // The complete new cache next to the old one: a failure leaves the old cache, its binding and size as they were.
    auto tensors = build(num_blocks, std::min(m_num_blocks, num_blocks), zero_initialize_new_blocks);
    try {
        bind_all(tensors);
    } catch (...) {
        bind_all(m_tensors);
        throw;
    }
    m_tensors = std::move(tensors);
    m_num_blocks = num_blocks;
}

void CacheController::deallocate(size_t group) {
    check_group(group);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_num_blocks == 0)
        return;
    // The rank requests hold the cache tensors too; the memory is freed only once they let go.
    bind_all(placeholders());
    m_tensors.clear();
    m_num_blocks = 0;
}

void CacheController::copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& block_copy_map) {
    check_group(group);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (block_copy_map.empty())
        return;
    OPENVINO_ASSERT(m_num_blocks > 0, "[TP] copy_blocks before the cache was allocated");
    for (const auto& [source, destinations] : block_copy_map) {
        OPENVINO_ASSERT(source < m_num_blocks, "[TP] copy_blocks source block ", source, " is outside the ", m_num_blocks,
                        " allocated blocks");
        for (const auto destination : destinations)
            OPENVINO_ASSERT(destination < m_num_blocks, "[TP] copy_blocks destination block ", destination,
                            " is outside the ", m_num_blocks, " allocated blocks");
    }
    for (const auto& rank_tensors : m_tensors) {
        for (const auto& tensor : rank_tensors) {
            const size_t block_bytes = tensor->get_strides().front();
            const auto block_shape = with_blocks(tensor->get_shape(), 1);
            auto& remote = as_remote(tensor);
            for (const auto& [source, destinations] : block_copy_map) {
                for (const auto destination : destinations)
                    remote.copy_to(tensor._ptr, source * block_bytes, destination * block_bytes, block_shape);
            }
        }
    }
}

void CacheController::zero_blocks(size_t group, const std::set<size_t>& block_indices) {
    check_group(group);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (block_indices.empty())
        return;
    OPENVINO_ASSERT(m_num_blocks > 0, "[TP] zero_blocks before the cache was allocated");
    OPENVINO_ASSERT(*block_indices.rbegin() < m_num_blocks, "[TP] zero_blocks block ", *block_indices.rbegin(),
                    " is outside the ", m_num_blocks, " allocated blocks");
    for (const auto& rank_tensors : m_tensors) {
        for (const auto& tensor : rank_tensors) {
            // Runs of consecutive blocks in one copy each.
            for (auto it = block_indices.begin(); it != block_indices.end();) {
                const size_t first = *it;
                size_t count = 1;
                while (++it != block_indices.end() && *it == first + count)
                    ++count;
                zero(tensor, first, count);
            }
        }
    }
}

uint64_t CacheController::register_request(const std::vector<ov::SoPtr<ov::IAsyncInferRequest>>& rank_requests) {
    std::vector<std::weak_ptr<ov::IAsyncInferRequest>> request;
    for (const auto& rank_request : rank_requests)
        request.push_back(rank_request._ptr);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_num_blocks > 0)
        bind(request, m_tensors);
    const auto id = m_next_request++;
    m_requests.emplace(id, std::move(request));
    return id;
}

void CacheController::unregister_request(uint64_t id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_requests.erase(id);
}

}  // namespace tp
}  // namespace ov
