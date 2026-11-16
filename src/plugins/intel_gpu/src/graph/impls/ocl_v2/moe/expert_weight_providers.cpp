// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <unordered_map>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

bool is_per_expert(const cldnn::layout& layout, size_t num_expert) {
    return layout.is_static() && layout.count() > 1 && !layout.get_shape().empty() && layout.get_shape()[0] == num_expert;
}

bool has_contiguous_expert_slices(const cldnn::layout& layout, size_t num_expert) {
    return cldnn::format::is_simple_data_format(layout.format) && layout.format.order().front() == 'b' && layout.bytes_count() % num_expert == 0;
}

cldnn::layout get_slot_layout(const cldnn::layout& input, size_t capacity, size_t num_expert) {
    OPENVINO_ASSERT(has_contiguous_expert_slices(input, num_expert), "OTD: expert slices of ", input.to_short_string(), " are not contiguous");
    auto slots_shape = input.get_shape();
    slots_shape[0] = capacity;
    return cldnn::layout(slots_shape, input.data_type, input.format);
}

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                                                      const RoutedWeightViews& /*inputs*/,
                                                                                      const RoutedWeightViews& /*slots*/,
                                                                                      cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert,
                                                 const RoutedWeightViews& /*inputs*/,
                                                 const RoutedWeightViews& /*slots*/,
                                                 cldnn::stream& /*stream*/) {
    return expert;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::filesystem::path weights_path)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _cache(std::make_shared<LRUCache>(capacity)),
      _weight_reader(_weights_path) {}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& resident) {
    _resident = &resident;
    _cache->set_initialized();
    _bound = true;
    GPU_DEBUG_TRACE << "MOE OTD bind: resident_slots=" << _capacity << " total_experts=" << _config.num_expert
                    << " shared_expert=" << (_config.num_shared_expert > 0 ? 1 : 0) << std::endl;
}

void OffloadExpertWeightProvider::fill_routed_weight_views(cldnn::moe_weights& /*weights*/, RoutedWeightViews& views) {
    // Offloaded: point at the LRU resident buffers (not the original weight inputs).
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    views.weight[0] = _resident->gate_w;
    views.scale[0] = _resident->gate_s;
    views.zp[0] = _resident->gate_z;
    views.weight[1] = _resident->up_w;
    views.scale[1] = _resident->up_s;
    views.zp[1] = _resident->up_z;
    views.weight[2] = _resident->down_w;
    views.scale[2] = _resident->down_s;
    views.zp[2] = _resident->down_z;
}

void OffloadExpertWeightProvider::reset_cache_if_slots_changed(const RoutedWeightViews& slots) {
    std::vector<const void*> handles;
    for (size_t i = 0; i < 3; i++) {
        for (const auto& mem : {slots.weight[i], slots.scale[i], slots.zp[i]}) {
            if (mem) {
                handles.push_back(mem->get_internal_params().mem);
            }
        }
    }
    if (handles != _slot_handles) {
        _slot_handles = std::move(handles);
        _cache = std::make_shared<LRUCache>(_capacity);
        _cache->set_initialized();
    }
}

void OffloadExpertWeightProvider::load(uint32_t expert, size_t slot, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) {
    cldnn::moe_weights slot_mems;
    slot_mems.gate_w = slots.weight[0];
    slot_mems.up_w = slots.weight[1];
    slot_mems.down_w = slots.weight[2];
    slot_mems.gate_s = slots.scale[0];
    slot_mems.up_s = slots.scale[1];
    slot_mems.down_s = slots.scale[2];
    slot_mems.gate_z = slots.zp[0];
    slot_mems.up_z = slots.zp[1];
    slot_mems.down_z = slots.zp[2];
    cldnn::moe_weights sources;
    sources.gate_s = slots.scale[0] ? inputs.scale[0] : nullptr;
    sources.up_s = slots.scale[1] ? inputs.scale[1] : nullptr;
    sources.down_s = slots.scale[2] ? inputs.scale[2] : nullptr;
    sources.gate_z = slots.zp[0] ? inputs.zp[0] : nullptr;
    sources.up_z = slots.zp[1] ? inputs.zp[1] : nullptr;
    sources.down_z = slots.zp[2] ? inputs.zp[2] : nullptr;
    moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, slot_mems, sources, {expert}, {slot});
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                                                     const RoutedWeightViews& inputs,
                                                                                     const RoutedWeightViews& slots,
                                                                                     cldnn::stream& stream) {
    reset_cache_if_slots_changed(slots);

    // Deduplicate to check capacity
    std::unordered_map<uint32_t, size_t> expert_to_slot;
    expert_to_slot.reserve(experts.size());

    std::vector<size_t> lease(experts.size());
    auto* perf = moe_otd::get_perf_counters();

    for (size_t i = 0; i < experts.size(); i++) {
        const uint32_t expert = experts[i];
        auto it = expert_to_slot.find(expert);
        if (it != expert_to_slot.end()) {
            lease[i] = it->second;
            continue;
        }

        // Check if we would exceed capacity
        if (expert_to_slot.size() >= _capacity) {
            return std::nullopt;
        }

        const auto item = _cache->get_lru_item(expert);
        const auto slot = item.first;

        if (item.second) {
            if (perf) {
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            if (perf) {
                perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
            }
            load(expert, slot, inputs, slots, stream);
            _cache->set_filled(slot);
        }

        expert_to_slot.emplace(expert, slot);
        lease[i] = slot;
    }

    return lease;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) {
    reset_cache_if_slots_changed(slots);
    auto* perf = moe_otd::get_perf_counters();

    const auto item = _cache->get_lru_item(expert);
    const auto slot = item.first;

    if (item.second) {
        if (perf) {
            perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (perf) {
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        load(expert, slot, inputs, slots, stream);
        _cache->set_filled(slot);
    }

    return slot;
}

}  // namespace ov::intel_gpu::ocl::moe
