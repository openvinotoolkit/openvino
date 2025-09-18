// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#include "openvino/runtime/allocator.hpp"

namespace intel_npu {

namespace utils {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
constexpr std::size_t BATCH_AXIS = 0;

struct AlignedAllocator {
public:
    AlignedAllocator(const size_t align_size) : _align_size(align_size) {}

    void* allocate(const size_t bytes, const size_t /*alignment*/) {
        return ::operator new(bytes, std::align_val_t(_align_size));
    }

    void deallocate(void* handle, const size_t /*bytes*/, const size_t /*alignment*/) noexcept {
        ::operator delete(handle, std::align_val_t(_align_size));
    }

    bool is_equal(const AlignedAllocator&) const {
        return true;
    }

private:
    const size_t _align_size;
};

static inline bool memory_and_size_aligned_to_standard_page_size(void* addr, size_t size) {
    auto addr_int = reinterpret_cast<uintptr_t>(addr);

    // addr is aligned to standard page size
    return (addr_int % STANDARD_PAGE_SIZE == 0) && (size % STANDARD_PAGE_SIZE == 0);
}

static inline size_t align_size_to_standard_page_size(size_t size) {
    return (size + utils::STANDARD_PAGE_SIZE - 1) & ~(utils::STANDARD_PAGE_SIZE - 1);
}

static inline std::shared_ptr<const ov::Model> exclude_model_ptr_from_map(ov::AnyMap& properties) {
    std::shared_ptr<const ov::Model> modelPtr = nullptr;
    if (properties.count(ov::hint::model.name())) {
        try {
            modelPtr = properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>();
        } catch (const ov::AssertFailure&) {
            try {
                modelPtr = std::const_pointer_cast<const ov::Model>(
                    properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>());
            } catch (const ov::Exception&) {
                OPENVINO_THROW("The value of the \"ov::hint::model\" configuration option (\"MODEL_PTR\") has the "
                               "wrong data type. Expected: std::shared_ptr<const ov::Model>.");
            }
        }
        properties.erase(ov::hint::model.name());
    }
    return modelPtr;
}

}  // namespace utils

}  // namespace intel_npu
