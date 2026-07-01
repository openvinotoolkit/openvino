// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace ov::intel_gpu::ocl::moe {

/// Holds the slot indices produced by a simultaneous acquisition.
/// Valid until release() is called (or provider destruction).
struct ExpertSlotLease {
    std::vector<uint32_t> slots;  // parallel to the input experts vector
};

/// Lightweight view of routed expert weight/scale/zp triplets for 3 GEMMs.
struct RoutedWeightViews {
    cldnn::memory::ptr weight[3];
    cldnn::memory::ptr scale[3];
    cldnn::memory::ptr zp[3];
};

// Abstraction over "where the weights of a MoE layer's experts live" and how they
// are made addressable on the device for a given set of activated experts.
//
// Two concrete strategies exist today, both expressible through this interface:
//   * Fully resident: every expert's weights stay in device memory for the whole
//     network lifetime (the ratio=0 path). acquire() is the identity mapping.
//   * Offloaded (OTD): expert weights live on disk and are streamed on demand into
//     a bounded set of device-resident slots managed by an LRU cache (ratio>0).
//     acquire() deduplicates, assigns/streams slots, and returns the slot per expert.
//
// Keeping this behind an interface lets the MoE execution paths be written once,
// independent of the residency strategy, and lets the offload machinery (LRU +
// disk streaming + capacity serialization) be reused by other plugins/back-ends
// by only swapping the device-buffer/stream adapter underneath.
class IExpertWeightProvider {
public:
    virtual ~IExpertWeightProvider() = default;

    // Maps a list of expert ids (as produced by routing, duplicates allowed) to
    // device-resident slot indices usable by the GEMM kernels.
    //
    // Contract: every slot returned for a single acquire() call is simultaneously
    // valid and mutually non-aliasing until the matching release(). The returned
    // vector is parallel to `experts` (experts[i] -> slot[i]); duplicate expert
    // ids map to the same slot. For the resident strategy this is the identity
    // mapping (slot == expert id). For the offloaded strategy the caller must not
    // request more unique experts than resident_capacity() in one acquire().
    virtual std::vector<uint32_t> acquire(const std::vector<uint32_t>& experts, cldnn::stream& stream) = 0;

    // Releases any pinning established by the most recent acquire(). Resident
    // providers have nothing to release.
    virtual void release() {}

    // Number of device-resident expert slots. 0 means fully resident (no bound on
    // how many experts can be addressed at once).
    virtual size_t resident_capacity() const = 0;

    // True when expert weights are streamed on demand rather than fully resident.
    virtual bool is_offloaded() const = 0;

    // --- New lease-based API (Phase 3) ---

    // Binds device-resident weight buffers. Must be called once before any acquire.
    // For resident providers this is a no-op. For offload providers this stores the
    // buffer pointers used for streaming.
    virtual void bind(cldnn::moe_weights& weights) {
        (void)weights;
    }

    // Returns true once bind() has been called (offloaded) or always true (resident).
    virtual bool is_bound() const {
        return true;
    }

    // Fills the RoutedWeightViews with the correct memory pointers for the 3 GEMMs.
    // Resident: points directly at the full expert buffers.
    // Offloaded: points at the LRU resident slot buffers.
    virtual void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) {
        views.weight[0] = weights.gate_w;
        views.scale[0] = weights.gate_s;
        views.zp[0] = weights.gate_z;
        views.weight[1] = weights.up_w;
        views.scale[1] = weights.up_s;
        views.zp[1] = weights.up_z;
        views.weight[2] = weights.down_w;
        views.scale[2] = weights.down_s;
        views.zp[2] = weights.down_z;
    }

    // Attempts to simultaneously acquire slots for all requested experts.
    // Returns nullopt if the number of unique experts exceeds resident_capacity()
    // (the caller should fall back to per-expert streaming).
    // The returned lease's slots vector is parallel to the input experts vector.
    virtual std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) = 0;

    // Acquires a single expert slot. Always succeeds (evicts LRU if needed).
    virtual uint32_t acquire_one(uint32_t expert, cldnn::stream& stream) = 0;

    // Releases a lease. Currently a no-op for both providers but kept for
    // future pinning / reference-count semantics.
    virtual void release(ExpertSlotLease& /*lease*/) {}
};

}  // namespace ov::intel_gpu::ocl::moe
