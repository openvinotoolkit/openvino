/* SPDX-License-Identifier: MIT */

/**
 * Copyright (c) 2025, Intel Corporation.
 */

/**
 * TODO: Some awesome documentation
 */

#pragma once

#include <stdio.h>
#include <cstdint>

namespace detail {

constexpr bool OPTIONAL = false;
constexpr bool REQUIRED = !OPTIONAL;

template < uint16_t id, bool policy = OPTIONAL >
struct CapabilityNo {
    static constexpr auto ID = id;
    static constexpr auto POLICY = policy;
};

}  // namespace

/**
 * Most like it's not required, but having compat namespace here
 * but it enables custom clang-tidy check for this code in compiler repo
 * as it gets imported there through ELF lib repo
 */
namespace compat {

#pragma pack(push, 1)

// what if multiple versions may pose problems while having multiple isCompatible implementations
// struct alignas(uint8_t) BatchSize : detail::CapabilityNo<0> {
//     explicit BatchSize(uint8_t mode) : mode(mode) {}
//     uint8_t mode;

//     static bool isCompatible(const BatchSize& blobMode) {
//         return blobMode.mode <= 4;
//     }
// };

struct alignas(uint8_t) DummyCapability : detail::CapabilityNo<0> {
    explicit DummyCapability(uint8_t mode) : mode(mode) {}
    uint8_t mode;

    static bool isCompatible(const DummyCapability& blobMode) {
        return blobMode.mode <= 4;
    }
};

#pragma pack(pop)

}  // namespace compat
