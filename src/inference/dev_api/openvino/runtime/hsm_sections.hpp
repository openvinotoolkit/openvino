// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Small per-tag section codecs that don't warrant a dedicated file of their own.
 * @file openvino/runtime/hsm_sections.hpp
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>

#include "openvino/runtime/hsm_format.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief Byte view of @p id, e.g. for #IWriter::add_section()'s view overload on a #model_id_tag section.
 * @param id Borrowed, never copied - the returned view stays valid only as long as @p id does.
 */
inline ov::util::MemoryView encode_model_id(const uint64_t& id) noexcept {
    return {reinterpret_cast<const std::byte*>(&id), sizeof(id)};
}

/**
 * @brief Recovers the id written by #encode_model_id().
 * @param section Inline payload, which always reads back as the full-width slot zero-filled past the bytes
 * actually written - so a fixed `sizeof(uint64_t)` is read here, never @p section's own length.
 * @return `std::nullopt` if @p section is too short to hold an id.
 */
inline std::optional<uint64_t> decode_model_id(ov::util::MemoryView section) noexcept {
    uint64_t id = 0;
    if (section.size() < sizeof(id)) {
        return std::nullopt;
    }
    std::memcpy(&id, section.data(), sizeof(id));
    return id;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
