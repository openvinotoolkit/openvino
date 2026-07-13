// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_otd_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "intel_gpu/runtime/engine.hpp"
#include "impls/ocl_v2/moe/moe_otd_runtime.hpp"

namespace cldnn {
namespace onednn {

// Transpose scale/zp from [OC, G] to [G, OC] layout for oneDNN consumption.
// This is the 2GEMM variant that computes OC from the actual per-expert memory size.
static void transpose_scale_or_zp(const layout& mem_layout,
                                  std::vector<uint8_t>& payload,
                                  size_t per_expert_size,
                                  size_t oc,
                                  size_t group_count) {
    const auto elem_type = ov::element::Type(mem_layout.data_type);
    const size_t elem_size = elem_type.bitwidth() / 8;
    if (elem_size == 0 || oc == 0 || group_count == 0)
        return;

    const size_t expected_size = oc * group_count * elem_size;
    if (expected_size != per_expert_size)
        return;  // Dimensions don't match, skip transpose

    std::vector<uint8_t> transposed(per_expert_size, 0);
    for (size_t o = 0; o < oc; o++) {
        for (size_t g = 0; g < group_count; g++) {
            const size_t src_elem_idx = o * group_count + g;
            const size_t dst_elem_idx = g * oc + o;
            std::memcpy(transposed.data() + dst_elem_idx * elem_size,
                        payload.data() + src_elem_idx * elem_size,
                        elem_size);
        }
    }
    payload.swap(transposed);
}

void Moe2GemmOtdContext::load_expert_from_disk(stream& s, bool is_up, size_t slot, uint32_t expert) {
    using namespace ov::intel_gpu::ocl;
    auto* perf = moe_otd::get_perf_counters();

    std::array<std::pair<size_t, bool>, 3> spec = is_up
        ? std::array<std::pair<size_t, bool>, 3>{{{UP_W, false}, {UP_S, true}, {UP_Z, true}}}
        : std::array<std::pair<size_t, bool>, 3>{{{DOWN_W, false}, {DOWN_S, true}, {DOWN_Z, true}}};
    std::array<memory::ptr, 3> mems = is_up
        ? std::array<memory::ptr, 3>{{resident.up_w, resident.up_s, resident.up_z}}
        : std::array<memory::ptr, 3>{{resident.down_w, resident.down_s, resident.down_z}};

    for (size_t t = 0; t < 3; t++) {
        auto& mem = mems[t];
        if (!mem)
            continue;
        const bool needs_transpose = spec[t].second;
        const size_t per_expert_size = mem->get_layout().bytes_count() / num_experts;
        const size_t base_offset = weight_bin_offsets[spec[t].first];
        const size_t src_offset = base_offset + expert * per_expert_size;
        const size_t dst_offset = slot * per_expert_size;

        std::vector<uint8_t> payload(per_expert_size);
        if (perf) {
            auto t0 = std::chrono::steady_clock::now();
            weight_reader->read(reinterpret_cast<char*>(payload.data()), per_expert_size, src_offset);
            auto t1 = std::chrono::steady_clock::now();
            perf->disk_io_ns.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                std::memory_order_relaxed);
        } else {
            weight_reader->read(reinterpret_cast<char*>(payload.data()), per_expert_size, src_offset);
        }

        if (needs_transpose) {
            const auto& shape = mem->get_layout().get_shape();
            const size_t tensor_oc = (shape.size() >= 2) ? shape[1] : 0;
            const size_t tensor_gc = (shape.size() >= 3) ? shape[2] : 0;
            if (perf) {
                auto t0 = std::chrono::steady_clock::now();
                transpose_scale_or_zp(mem->get_layout(), payload, per_expert_size, tensor_oc, tensor_gc);
                auto t1 = std::chrono::steady_clock::now();
                perf->transpose_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                    std::memory_order_relaxed);
            } else {
                transpose_scale_or_zp(mem->get_layout(), payload, per_expert_size, tensor_oc, tensor_gc);
            }
        }

        if (perf) {
            auto t0 = std::chrono::steady_clock::now();
            mem->copy_from(s, payload.data(), 0, dst_offset, per_expert_size, true);
            auto t1 = std::chrono::steady_clock::now();
            perf->gpu_copy_ns.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                std::memory_order_relaxed);
            perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            mem->copy_from(s, payload.data(), 0, dst_offset, per_expert_size, true);
        }
    }
}

std::vector<size_t> Moe2GemmOtdContext::load_experts(stream& s, const std::vector<uint32_t>& expert_ids, bool is_up) {
    using namespace ov::intel_gpu::ocl;

    // expert_ids must be sorted and unique (caller guarantees this).
    // The grouped matmul reads weight slot g for token-group g, and token groups are laid
    // out in sorted-expert order, so expert_ids[i] MUST end up physically at slot i for the
    // slots consumed by this matmul call. Slots [n, capacity) act as an LRU cache that keeps
    // previously loaded experts resident across tokens: on a subsequent token an expert that
    // is still resident is moved into place with a GPU->GPU copy (no disk/transpose/upload),
    // and experts displaced from the active region are parked into free cache slots instead
    // of being evicted. This is what makes decode fast (mirrors the 3GEMM LRU behaviour while
    // respecting the grouped-matmul contiguity constraint).
    const size_t n = std::min(expert_ids.size(), capacity);
    std::vector<size_t> slots(expert_ids.size());
    for (size_t i = 0; i < slots.size(); i++)
        slots[i] = i;  // group g always reads slot g

    auto* perf = moe_otd::get_perf_counters();
    auto& slot_contents = is_up ? slot_contents_up : slot_contents_down;
    auto& swap_tmp = is_up ? swap_tmp_up : swap_tmp_down;

    // Collect the valid (non-null) tensors for this direction together with their per-expert
    // byte size, so weight/scale/zp are always moved/loaded together and stay consistent.
    struct TInfo {
        size_t offset_idx;
        memory* mem;
        bool needs_transpose;
        size_t per_expert_size;
    };
    std::array<std::pair<size_t, bool>, 3> spec = is_up
        ? std::array<std::pair<size_t, bool>, 3>{{{UP_W, false}, {UP_S, true}, {UP_Z, true}}}
        : std::array<std::pair<size_t, bool>, 3>{{{DOWN_W, false}, {DOWN_S, true}, {DOWN_Z, true}}};
    std::array<memory::ptr, 3> mems = is_up
        ? std::array<memory::ptr, 3>{{resident.up_w, resident.up_s, resident.up_z}}
        : std::array<memory::ptr, 3>{{resident.down_w, resident.down_s, resident.down_z}};

    std::vector<TInfo> tensors;
    tensors.reserve(3);
    size_t max_expert_size = 0;
    for (size_t t = 0; t < 3; t++) {
        if (!mems[t])
            continue;
        const size_t per_expert_size = mems[t]->get_layout().bytes_count() / num_experts;
        tensors.push_back(TInfo{spec[t].first, mems[t].get(), spec[t].second, per_expert_size});
        max_expert_size = std::max(max_expert_size, per_expert_size);
    }
    if (tensors.empty())
        return slots;

    // Lazily allocate the GPU->GPU swap scratch (one expert slot of the largest tensor).
    if (!swap_tmp || swap_tmp->get_layout().bytes_count() < max_expert_size) {
        auto* eng = tensors.front().mem->get_engine();
        auto tmp_layout = cldnn::layout(ov::Shape{max_expert_size}, cldnn::data_types::u8, cldnn::format::bfyx);
        swap_tmp = eng->allocate_memory(tmp_layout, cldnn::allocation_type::usm_device, false);
    }

    // GPU->GPU move of one expert slot (all tensors) from -> to (overwrites destination).
    // Blocking to match the proven-correct disk-upload path: the Level Zero queue may route
    // copies to a dedicated copy engine, so we synchronize rather than rely on in-order
    // ordering relative to the following matmul.
    auto dev_move = [&](size_t from, size_t to) {
        for (auto& t : tensors)
            t.mem->copy_from(s, *t.mem, from * t.per_expert_size, to * t.per_expert_size, t.per_expert_size, true);
    };
    // GPU->GPU swap of two expert slots (all tensors) via the scratch buffer.
    auto dev_swap = [&](size_t a, size_t b) {
        for (auto& t : tensors) {
            swap_tmp->copy_from(s, *t.mem, a * t.per_expert_size, 0, t.per_expert_size, true);
            t.mem->copy_from(s, *t.mem, b * t.per_expert_size, a * t.per_expert_size, t.per_expert_size, true);
            t.mem->copy_from(s, *swap_tmp, 0, b * t.per_expert_size, t.per_expert_size, true);
        }
    };

    // Set of experts required by this call (used to protect them from cache eviction).
    std::unordered_set<uint32_t> needed(expert_ids.begin(), expert_ids.begin() + n);
    // Current location of every resident expert: expert_id -> slot.
    std::unordered_map<uint32_t, size_t> loc;
    for (size_t sl = 0; sl < capacity; sl++)
        if (slot_contents[sl] >= 0)
            loc.emplace(static_cast<uint32_t>(slot_contents[sl]), sl);

    // Pick a cache slot in [n, capacity) to park a displaced expert: prefer an empty slot,
    // otherwise the first slot holding an expert not needed by this call. Returns SIZE_MAX
    // when the cache region is full of still-needed experts (only possible when n == capacity).
    auto pick_parking = [&]() -> size_t {
        for (size_t p = n; p < capacity; p++)
            if (slot_contents[p] < 0)
                return p;
        for (size_t p = n; p < capacity; p++)
            if (!needed.count(static_cast<uint32_t>(slot_contents[p])))
                return p;
        return std::numeric_limits<size_t>::max();
    };

    for (size_t i = 0; i < n; i++) {
        const uint32_t expert = expert_ids[i];

        // Already in the right slot: pure cache hit, nothing to do.
        if (slot_contents[i] == static_cast<int32_t>(expert)) {
            if (perf)
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        auto it = loc.find(expert);
        if (it != loc.end()) {
            // Resident elsewhere: swap it into slot i (GPU->GPU, no disk).
            const size_t sl = it->second;
            dev_swap(i, sl);
            const int32_t displaced = slot_contents[i];
            slot_contents[i] = static_cast<int32_t>(expert);
            slot_contents[sl] = displaced;
            loc[expert] = i;
            if (displaced >= 0)
                loc[static_cast<uint32_t>(displaced)] = sl;
            if (perf)
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Cache miss: expert must be loaded from disk into slot i. Preserve the expert
        // currently occupying slot i by parking it into a free cache slot when possible.
        const int32_t displaced = slot_contents[i];
        if (displaced >= 0) {
            const size_t park = pick_parking();
            if (park != std::numeric_limits<size_t>::max()) {
                const int32_t evicted = slot_contents[park];
                dev_move(i, park);
                slot_contents[park] = displaced;
                loc[static_cast<uint32_t>(displaced)] = park;
                if (evicted >= 0)
                    loc.erase(static_cast<uint32_t>(evicted));
            } else {
                loc.erase(static_cast<uint32_t>(displaced));
            }
        }

        if (perf)
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
        load_expert_from_disk(s, is_up, i, expert);
        slot_contents[i] = static_cast<int32_t>(expert);
        loc[expert] = i;
    }

    return slots;
}

std::vector<size_t> Moe2GemmOtdContext::acquire_experts_lru(stream& s, const std::vector<uint32_t>& expert_ids, bool is_up) {
    using namespace ov::intel_gpu::ocl;
    auto* perf = moe_otd::get_perf_counters();
    auto& slot_contents = is_up ? slot_contents_up : slot_contents_down;
    auto& slot_used = is_up ? slot_used_up : slot_used_down;

    std::vector<size_t> slots(expert_ids.size());

    // Current residency: expert_id -> slot.
    std::unordered_map<uint32_t, size_t> loc;
    for (size_t sl = 0; sl < capacity; sl++)
        if (slot_contents[sl] >= 0)
            loc.emplace(static_cast<uint32_t>(slot_contents[sl]), sl);

    // Experts needed by this call must not be evicted to make room for one another.
    std::unordered_set<uint32_t> needed(expert_ids.begin(), expert_ids.end());

    // Choose a slot for a miss: prefer an empty slot, otherwise evict the least-recently-used
    // slot whose expert is not needed by this call.
    auto pick_victim = [&]() -> size_t {
        for (size_t sl = 0; sl < capacity; sl++)
            if (slot_contents[sl] < 0)
                return sl;
        size_t victim = std::numeric_limits<size_t>::max();
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t sl = 0; sl < capacity; sl++) {
            if (needed.count(static_cast<uint32_t>(slot_contents[sl])))
                continue;
            if (slot_used[sl] < oldest) {
                oldest = slot_used[sl];
                victim = sl;
            }
        }
        return victim;
    };

    for (size_t i = 0; i < expert_ids.size(); i++) {
        const uint32_t expert = expert_ids[i];

        auto it = loc.find(expert);
        if (it != loc.end()) {
            // Hit: reuse the resident slot, no copy.
            const size_t sl = it->second;
            slots[i] = sl;
            slot_used[sl] = ++lru_tick;
            if (perf)
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Miss: load into a free/evicted slot (no relocation of other experts).
        const size_t sl = pick_victim();
        OPENVINO_ASSERT(sl != std::numeric_limits<size_t>::max(),
                        "acquire_experts_lru: no slot available (active experts exceed capacity)");
        const int32_t evicted = slot_contents[sl];
        if (evicted >= 0)
            loc.erase(static_cast<uint32_t>(evicted));

        if (perf)
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
        load_expert_from_disk(s, is_up, sl, expert);
        slot_contents[sl] = static_cast<int32_t>(expert);
        slot_used[sl] = ++lru_tick;
        loc[expert] = sl;
        slots[i] = sl;
    }

    return slots;
}

}  // namespace onednn
}  // namespace cldnn
