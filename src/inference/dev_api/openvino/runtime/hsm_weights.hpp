// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

using WeightsBuffer = std::vector<std::byte>;

/**
 * @brief Adds @p weights as the Core-owned #weights_tag section, as an unchanged byte copy made by the writer.
 * @param writer Writer receiving the section.
 * @param weights Raw weights bytes; borrowed, keep alive until #IWriter::finalize().
 * @param align Alignment requirements for the pointer-mode payload.
 */
OPENVINO_RUNTIME_API bool encode_weights(IWriter& writer, ov::util::MemoryView weights, SectionAlignment align = {});

/**
 * @brief Decodes the Core-owned weights section into an unchanged owned byte buffer.
 * @return The weights bytes, or `std::nullopt` when the section is absent or cannot be read.
 */
OPENVINO_RUNTIME_API std::optional<WeightsBuffer> decode_weights(const Reader& reader);

/**
 * @brief Decodes the Core-owned weights section into a non-owning memory view.
 * @return The weights view, or `std::nullopt` when the section is absent or cannot be read.
 */
OPENVINO_RUNTIME_API std::optional<ov::util::MemoryView> decode_weights_view(const Reader& reader);

}  // namespace v1
}  // namespace ov::runtime::hsm
