// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/mmap_object.hpp"

namespace ov {

const util::MemoryProperties& MappedMemory::get_properties() const noexcept {
    static const util::MemoryProperties instance{};
    return instance;
}

void MappedMemory::hint_evict() noexcept {
    hint_evict(0, auto_size);
}

// A hint may always no-op; the ranged, stateful hint_prefetch(offset, size) is the real API.
void MappedMemory::hint_prefetch() const {}

}  // namespace ov
