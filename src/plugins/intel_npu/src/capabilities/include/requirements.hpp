//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <stdio.h>

#include <cstdint>
#include <vector>

#include "meta.hpp"
#include "openvino/core/layout.hpp"

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

struct alignas(uint64_t) Expression : meta::CapabilityNo<0, meta::REQUIRED> {
    explicit Expression(uint64_t expression) : expression(expression) {}

    uint64_t expression;

    bool isCompatible(const Expression& expression) const {
        // do the checks here? or in check.cpp?
        return true;
    }
};

struct alignas(uint64_t) ELFBlob : meta::CapabilityNo<1, meta::REQUIRED> {
    // data should be an uint8_t* instead
    explicit ELFBlob(uint64_t size, uint64_t data) : size(size), data(data) {}

    uint64_t size;
    uint64_t data;
    // uint8_t* data;

    // to revisit: its interesting that without the const qualifier, the build fails with
    // error: static assertion failed due to requirement 'meta::IsCapability<compat::ELFBlob, void>::value': Type does not satisfy Capability concept
    bool isCompatible(const ELFBlob& blob) const {
        // TODO: is this needed here?
        return true;
    }
};

struct alignas(uint64_t) WeightsSeparationRequirement : meta::CapabilityNo<2, meta::REQUIRED> {
    // explicit WeightsSeparationRequirement(uint64_t size, std::vector<uint64_t> initSizes) : size(size),
    // initSizes(initSizes) {}

    explicit WeightsSeparationRequirement(uint64_t size, uint64_t initSizes) : size(size), initSizes(initSizes) {}

    uint64_t size;  // in bytes
    // std::vector<uint64_t> initSizes;
    uint64_t initSizes;

    // whatever check is gonna be here
    bool isCompatible(const WeightsSeparationRequirement& blobValue) const {
        printf("!!!     compat::WeightlessBlob::isCompatible !!!\n");
        return blobValue.size <= size;
    }
};

struct alignas(int64_t) BatchSize : meta::CapabilityNo<3, meta::REQUIRED> {
    explicit BatchSize(int64_t size) : size(size) {}

    int64_t size;

    bool isCompatible(const BatchSize& blobMode) const {
        return blobMode.size <= 4;
    }
};

// TODO: is there a check or a way to check if there is a duplicate capability ID?
// the real alignment would be uint8_t?
struct alignas(uint64_t) InputOutputLayouts : meta::CapabilityNo<4, meta::REQUIRED> {
    // explicit InputOutputLayouts(std::vector<ov::Layout> inputLayouts, std::vector<ov::Layout> outputLayouts) :
    // inputLayouts(inputLayouts), outputLayouts(outputLayouts) {}

    // std::vector<ov::Layout> inputLayouts;
    // std::vector<ov::Layout> outputLayouts;

    explicit InputOutputLayouts(uint64_t inputLayouts, uint64_t outputLayouts)
        : inputLayouts(inputLayouts),
          outputLayouts(outputLayouts) {}

    uint64_t inputLayouts;
    uint64_t outputLayouts;

    bool isCompatible(const InputOutputLayouts& ioLayouts) const {
        return true;
    }
};

#pragma pack(pop)

}  // namespace compat
