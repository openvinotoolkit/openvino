// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Small per-tag section codecs that don't warrant a dedicated file of their own.
 * @file openvino/runtime/hsm_model_id_codec.hpp
 */

#pragma once

#include <cstdint>
#include <optional>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

// No encoder needed: #model_id_tag is inline, so the id goes straight into #IWriter::add_section() as
// `ov::util::as_memory_view(id)`.

/**
 * @brief Reads the model id from a #model_id_tag section.
 * @note Layout: bytes [0, 8) hold the id as a native-byte-order `uint64_t`; the rest of the inline slot is
 * ignored.
 * @param section Section to decode; its tag must be #model_id_tag.
 * @return `std::nullopt` if the read fails.
 */
OPENVINO_RUNTIME_API std::optional<uint64_t> decode_model_id(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
