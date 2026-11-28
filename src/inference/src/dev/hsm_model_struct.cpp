// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include "openvino/runtime/shared_buffer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
std::optional<std::shared_ptr<ov::AlignedBuffer>> decode_model_struct(const Section& section) {
    // Section::read() would fail for a section without a source and memcpy from null for an empty MemoryView.
    if (section.size() == 0) {
        return std::make_shared<ov::AlignedBuffer>();  // AlignedBuffer(0) would hold one byte
    }
    if (const auto view = section.view(); !view.empty()) {
        // Non-owning, read-only wrap: the container's owner keeps the bytes alive.
        return std::make_shared<ov::SharedBuffer<std::nullptr_t>>(const_cast<std::byte*>(view.data()),
                                                                  view.size(),
                                                                  nullptr);
    }
    auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
    return section.read(buffer->get_ptr()) ? std::optional(std::move(buffer)) : std::nullopt;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
