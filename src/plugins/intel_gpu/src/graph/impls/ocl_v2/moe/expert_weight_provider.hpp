// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "intel_gpu/runtime/stream.hpp"

namespace ov::intel_gpu::ocl::moe {

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
    virtual std::vector<uint32_t> acquire(size_t layer,
                                          const std::vector<uint32_t>& experts,
                                          cldnn::stream& stream) = 0;

    // Releases any pinning established by the most recent acquire(). Resident
    // providers have nothing to release.
    virtual void release() {}

    // Number of device-resident expert slots. 0 means fully resident (no bound on
    // how many experts can be addressed at once).
    virtual size_t resident_capacity() const = 0;

    // True when expert weights are streamed on demand rather than fully resident.
    virtual bool is_offloaded() const = 0;
};

}  // namespace ov::intel_gpu::ocl::moe
