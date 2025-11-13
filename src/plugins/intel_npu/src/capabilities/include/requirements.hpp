//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <stdio.h>
#include <cstdint>

#include "meta.hpp"

namespace compat {

// a list of "requirements" types for compiler to serialize in a blob
// to indicate what a given blob "requires" from the platform to support it
// serialization/de-serialization is implemented via reinterpret_cast
// for simplicity and efficiency, so binary-compatibility of these types
// to be preserved even across different compilers/platforms
// (e.g. x86 vs RISC-V); it is not a new use-case, as MappedInference types
// already follow that design
//
// to keep ABI stable following condition must be ensured:
// 1) the same endianess is used across compilers/platforms
// 2) explicit padding; see NNRT API headers for guidelines
// 3) explicit alignment; see NNRT API headers for guidelines
// 4) only standard types with fixed-size as non-static data-member variables
// 5) standard layout types, e.g. no virtual table pointer, all non-static
//    data-member variables have the same access specifier
// 6) types are trivially-copyable
// 7) types are POD: no pointer or reference non-static data-member variables
//
// 1) is to be checked at built-time via CMake; note that a solution with C++
// code via union and type-punning (e.g. proposed in https://stackoverflow.com/a/1001373)
// exhibits undefined behavior as while C permits it, C++ doesn't
// (see https://stackoverflow.com/a/11996970); project won't build if all systems involved
// are little-endian
//
// 2) is guaranteed by usage of "#pragma pack(push, 1)" and verified by custom clang-tidy
// check (on compiler side)
//
// 3) is guaranteed by usage of "NPU_ALIGNMENT" and verified by custom clang-tidy check
// (on compiler side)
//
// 4) verified by custom clang-tidy check (on compiler side)
//
// 5) is guaranteed by compat::meta::IsRequirement (requires compat::meta::IsMemCopyable,
// that requires std::is_standard_layout)
//
// 6) is guaranteed by compat::meta::IsRequirement (requires compat::meta::IsMemCopyable,
// that requires std::is_trivially_copyable)
//
// 7) verified by custom clang-tidy check (on compiler side)

#pragma pack(push, 1)

struct alignas(uint16_t) WeightlessBlob : meta::CapabilityNo<0, meta::REQUIRED> {
    explicit WeightlessBlob(uint16_t value) : value(value) {}
    uint16_t value;

    bool isCompatible(const WeightlessBlob& blobValue) const {
        printf("!!!     compat::WeightlessBlob::isCompatible !!!\n");
        return blobValue.value <= value;
    }
};

#pragma pack(pop)

}  // namespace compat
