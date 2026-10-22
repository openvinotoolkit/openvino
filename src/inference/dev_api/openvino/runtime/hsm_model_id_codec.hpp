// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Small per-tag section codecs that don't warrant a dedicated file of their own.
 * @file openvino/runtime/hsm_model_id_codec.hpp
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>

#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief Recovers the id written by #wrap_model_id().
 * @param section Section whose #Section::view() is read.
 * @return `std::nullopt` if @p section's tag isn't #model_id_tag.
 */
inline std::optional<uint64_t> decode_model_id(const Section& section) noexcept {
    if (section.tag().value != model_id_tag.value) {
        return std::nullopt;
    }

    uint64_t id{};
    if (section.read(0, sizeof(id), reinterpret_cast<std::byte*>(&id))) {
        return id;
    }

    return std::nullopt;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
