// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include "openvino/runtime/shared_buffer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
std::shared_ptr<ov::AlignedBuffer> decode_model_struct(const Section& section, std::shared_ptr<ov::AlignedBuffer> owner) {
    // Section::read() would fail for a section without a source and memcpy from null for an empty MemoryView.
    if (section.size() == 0) {
        return std::make_shared<ov::AlignedBuffer>();  // AlignedBuffer(0) would hold one byte
    }
    if (const auto view = section.view(); owner && !view.empty()) {
        // Read-only wrap; the buffer is never written through.
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(const_cast<std::byte*>(view.data()),
                                                                                    view.size(),
                                                                                    std::move(owner));
    }
    auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
    return section.read(buffer->get_ptr()) ? buffer : nullptr;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
