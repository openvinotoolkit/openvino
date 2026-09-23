// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_format.hpp"

#include <algorithm>

namespace ov::runtime {

const HSMHeader& HSMContainerView::header() const noexcept {
    return HSMHeader::view(reinterpret_cast<const uint8_t*>(begin()));
}

const ManifestEntry& HSMContainerView::manifest() const noexcept {
    return *reinterpret_cast<const ManifestEntry*>(begin() + header().manifest_offset);
}

bool HSMContainerView::validate() const noexcept {
    if (static_cast<size_t>(end() - begin()) < sizeof(HSMHeader)) {
        return false;
    }
    const auto& hdr = header();
    if (!is_valid_header_fields(hdr) || hdr.container_size > size()) {
        return false;
    }

    if (hdr.manifest_size == 0) {
        return true;
    }

    const auto* entries = &manifest();
    return std::all_of(entries, entries + manifest_count(), [&hdr](const ManifestEntry& entry) {
        return is_valid_section_bounds(entry, hdr);
    });
}

size_t HSMMultiBlobView::blob_count() const noexcept {
    auto view = m_view;
    size_t count = 0;
    while (view.size() >= sizeof(HSMHeader)) {
        const auto next = advance_container(view);
        if (!next) {
            break;
        }
        count += next->is_blob() ? 1 : 0;
        view = next->remaining();
    }
    return count;
}

HSMContainerView HSMMultiBlobView::blob_at(size_t index) const noexcept {
    auto view = m_view;
    while (view.size() >= sizeof(HSMHeader)) {
        const auto next = advance_container(view);
        if (!next) {
            break;
        }
        if (next->is_blob()) {
            if (index == 0) {
                return HSMContainerView{view.data(), next->container_size()};
            }
            --index;
        }
        view = next->remaining();
    }
    return {};
}

std::optional<HSMMultiBlobView::NextContainer> HSMMultiBlobView::advance_container(
    const ov::util::MemoryView& view) noexcept {
    const auto& hdr = HSMHeader::view(reinterpret_cast<const uint8_t*>(view.data()));
    if (!is_recognized_header(hdr) || hdr.container_size > view.size()) {
        return std::nullopt;
    } else {
        const auto container_size = static_cast<size_t>(hdr.container_size);
        return NextContainer{ov::util::MemoryView{view.data() + container_size, view.size() - container_size},
                             container_size,
                             hdr.magic == BlobMagic::single};
    }
}

}  // namespace ov::runtime
