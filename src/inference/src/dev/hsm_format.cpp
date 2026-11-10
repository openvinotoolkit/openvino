// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_format.hpp"

#include <algorithm>

namespace ov::runtime::hsm {

MultiBlobView::NextContainer::NextContainer(ov::util::MemoryView remaining,
                                            size_t container_size,
                                            bool is_blob) noexcept
    : m_remaining(remaining),
      m_container_size(container_size),
      m_is_blob(is_blob) {}

const ov::util::MemoryView& MultiBlobView::NextContainer::remaining() const noexcept {
    return m_remaining;
}

size_t MultiBlobView::NextContainer::container_size() const noexcept {
    return m_container_size;
}

bool MultiBlobView::NextContainer::is_blob() const noexcept {
    return m_is_blob;
}

const Header& ContainerView::header() const noexcept {
    return Header::view(m_view);
}

const ManifestEntry& ContainerView::manifest() const noexcept {
    return *reinterpret_cast<const ManifestEntry*>(begin() + header().manifest_offset);
}

bool ContainerView::validate() const noexcept {
    if (static_cast<size_t>(end() - begin()) < sizeof(Header)) {
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
    return std::all_of(entries, entries + manifest_count(), [&hdr](const auto& entry) {
        return is_valid_section_bounds(entry, hdr);
    });
}

size_t MultiBlobView::blob_count() const noexcept {
    auto view = m_view;
    size_t count = 0;
    while (view.size() >= sizeof(Header)) {
        const auto next = advance_container(view);
        if (!next) {
            break;
        }
        count += next->is_blob() ? 1 : 0;
        view = next->remaining();
    }
    return count;
}

ContainerView MultiBlobView::blob_at(size_t index) const noexcept {
    auto view = m_view;
    while (view.size() >= sizeof(Header)) {
        const auto next = advance_container(view);
        if (!next) {
            break;
        }
        if (next->is_blob()) {
            if (index == 0) {
                return ContainerView{view.data(), next->container_size()};
            }
            --index;
        }
        view = next->remaining();
    }
    return {};
}

std::optional<MultiBlobView::NextContainer> MultiBlobView::advance_container(
    const ov::util::MemoryView& view) noexcept {
    const auto& hdr = Header::view(view);
    if (!is_recognized_header(hdr) || hdr.container_size > view.size()) {
        return std::nullopt;
    } else {
        const auto container_size = static_cast<size_t>(hdr.container_size);
        return NextContainer{ov::util::MemoryView{view.data() + container_size, view.size() - container_size},
                             container_size,
                             hdr.magic == BlobMagic::single};
    }
}

}  // namespace ov::runtime::hsm
