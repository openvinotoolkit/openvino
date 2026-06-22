// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <unordered_map>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in, matching
// how moe_3gemm_swiglu_opt.cpp includes the runtime header.
#include "../primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

std::vector<uint32_t> ResidentExpertWeightProvider::acquire(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    // Fully resident: the expert id is already the addressable slot.
    return experts;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::string weights_path,
                                                         size_t layer_index)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _layer_index(layer_index),
      _cache(std::make_shared<LRUCache>(capacity)) {}

void OffloadExpertWeightProvider::bind_resident_buffers(cldnn::moe_weights& resident) {
    _resident = &resident;
    _cache->m_initialized = true;
}

std::vector<uint32_t> OffloadExpertWeightProvider::acquire(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    std::vector<uint32_t> slots(experts.size());

    // Deduplicate while preserving first-seen order so the LRU eviction order is
    // identical to the legacy per-expert remap path.
    std::unordered_map<uint32_t, uint32_t> expert_to_slot;
    expert_to_slot.reserve(experts.size());

    auto* perf = moe_otd::get_perf_counters();

    for (size_t i = 0; i < experts.size(); i++) {
        const uint32_t expert = experts[i];
        auto it = expert_to_slot.find(expert);
        if (it != expert_to_slot.end()) {
            slots[i] = it->second;
            continue;
        }

        const auto item = _cache->get_lru_item(_layer_index, expert);
        OPENVINO_ASSERT(item.first <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "LRU slot index overflow: ", item.first);
        const auto slot = static_cast<uint32_t>(item.first);

        if (item.second) {
            if (perf)
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (perf)
                perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
            OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound before acquire()");
            moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weights_path, *_resident, {expert}, {slot}, _layer_index);
            _cache->set_filled(slot);
        }

        expert_to_slot.emplace(expert, slot);
        slots[i] = slot;
    }

    return slots;
}

}  // namespace ov::intel_gpu::ocl::moe
