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

/// Slot indices produced by a simultaneous acquisition (parallel to the input experts vector).
using ExpertSlotLease = std::vector<size_t>;

/// Lightweight view of routed expert weight/scale/zp triplets for 3 GEMMs.
struct RoutedWeightViews {
    cldnn::memory::ptr weight[3];
    cldnn::memory::ptr scale[3];
    cldnn::memory::ptr zp[3];
};

// A scale / zero point shared by all the experts (scalar) is used as is, it has no per-expert slices
bool is_per_expert(const cldnn::layout& layout, size_t num_expert);

// Whether the slice of each expert is one contiguous byte range, which the slot copies require
bool has_contiguous_expert_slices(const cldnn::layout& layout, size_t num_expert);

// Layout of the slot buffer of an input with a slice per expert: the same slice, `capacity` times
cldnn::layout get_slot_layout(const cldnn::layout& input, size_t capacity, size_t num_expert);

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

    // Releases any pinning established by the most recent acquisition.
    virtual void release() = 0;

    // Number of device-resident expert slots. 0 means fully resident (no bound on
    // how many experts can be addressed at once).
    virtual size_t resident_capacity() const = 0;

    // True when expert weights are streamed on demand rather than fully resident.
    virtual bool is_offloaded() const = 0;

    // Attempts to simultaneously acquire slots for all requested experts.
    // Returns nullopt if the number of unique experts exceeds resident_capacity()
    // (the caller should fall back to per-expert streaming).
    // The returned lease's slots vector is parallel to the input experts vector.
    // `inputs` are the primitive inputs, `slots` the slot buffers that a miss loads; a scale / zero point without a slot buffer is
    // shared by all the experts.
    virtual std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                                    const RoutedWeightViews& inputs,
                                                                    const RoutedWeightViews& slots,
                                                                    cldnn::stream& stream) = 0;

    // Acquires a single expert slot. Always succeeds (evicts LRU if needed).
    virtual size_t acquire_one(uint32_t expert, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) = 0;

    // Binds device-resident weight buffers. Called once on first execution.
    virtual void bind(cldnn::moe_weights& resident) = 0;

    // Returns true after bind() has been called.
    virtual bool is_bound() const = 0;
};

}  // namespace ov::intel_gpu::ocl::moe
