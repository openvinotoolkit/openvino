// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

namespace ov {
namespace {
thread_local MMapConstantsConfig mmap_constants_config;
}  // namespace

const MMapConstantsConfig& get_mmap_constants_config() {
    return mmap_constants_config;
}

ScopedMMapConstantsConfig::ScopedMMapConstantsConfig(const MMapConstantsConfig& config)
    : m_previous_config{mmap_constants_config} {
    mmap_constants_config = config;
}

ScopedMMapConstantsConfig::~ScopedMMapConstantsConfig() {
    mmap_constants_config = m_previous_config;
}

bool TemporaryFileBackedAllocator::is_equal(const TemporaryFileBackedAllocator&) const {
    return true;
}

#if !defined(_WIN32) && !defined(__unix__) && !defined(__APPLE__)
void* TemporaryFileBackedAllocator::allocate(size_t, size_t) {
    OPENVINO_THROW("Temporary mmap constant storage is not supported on this platform");
}

void TemporaryFileBackedAllocator::deallocate(void*, size_t, size_t) noexcept {}
#endif

}  // namespace ov
