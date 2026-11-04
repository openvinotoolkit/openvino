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

namespace {
// A zero point shared by all the experts (scalar) is used as is, it has no per-expert slices
bool is_per_expert(const cldnn::memory::ptr& mem, size_t num_expert) {
    if (!mem) {
        return false;
    }
    const auto& layout = mem->get_layout();
    return layout.is_static() && layout.count() > 1 && !layout.get_shape().empty() && layout.get_shape()[0] == num_expert;
}

cldnn::memory::ptr allocate_slots(const cldnn::memory::ptr& mem, size_t capacity, size_t num_expert) {
    const auto& layout = mem->get_layout();
    OPENVINO_ASSERT(cldnn::format::is_simple_data_format(layout.format) && layout.format.order().front() == 'b' &&
                        layout.bytes_count() % num_expert == 0,
                    "OTD: expert slices of ",
                    layout.to_short_string(),
                    " are not contiguous");
    auto& engine = *mem->get_engine();
    auto slots_shape = layout.get_shape();
    slots_shape[0] = capacity;
    auto slots_mem = engine.allocate_memory(cldnn::layout(slots_shape, layout.data_type, layout.format), engine.get_preferred_memory_allocation_type(), false);
    // Viewed with the full layout, the same way as the partially uploaded weights
    return engine.reinterpret_buffer(*slots_mem, layout);
}
}  // namespace

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& /*stream*/) {
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
    if (_bound) {
        // Another instance sharing this provider uses the same slot buffers
        if (&resident != _resident) {
            resident = *_resident;
        }
        return;
    }
    const auto num_expert = static_cast<size_t>(_config.num_expert);
    for (auto member : {&cldnn::moe_weights::gate_s,
                        &cldnn::moe_weights::up_s,
                        &cldnn::moe_weights::down_s,
                        &cldnn::moe_weights::gate_z,
                        &cldnn::moe_weights::up_z,
                        &cldnn::moe_weights::down_z}) {
        auto& mem = resident.*member;
        if (is_per_expert(mem, num_expert)) {
            _sources.*member = mem;
            mem = allocate_slots(mem, _capacity, num_expert);
        }
    }
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

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    // Deduplicate to check capacity
    std::unordered_map<uint32_t, size_t> expert_to_slot;
    expert_to_slot.reserve(experts.size());

    std::vector<size_t> slots(experts.size());
    auto* perf = moe_otd::get_perf_counters();

    for (size_t i = 0; i < experts.size(); i++) {
        const uint32_t expert = experts[i];
        auto it = expert_to_slot.find(expert);
        if (it != expert_to_slot.end()) {
            slots[i] = it->second;
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
            OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
            moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, _sources, {expert}, {slot});
            _cache->set_filled(slot);
        }

        expert_to_slot.emplace(expert, slot);
        slots[i] = slot;
    }

    return slots;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& stream) {
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
        OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
        moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, _sources, {expert}, {slot});
        _cache->set_filled(slot);
    }

    return slot;
}

}  // namespace ov::intel_gpu::ocl::moe
