// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

using WeightsBuffer = std::vector<std::byte>;

/**
 * @brief Produces raw weights bytes for a #weights_tag section on demand.
 * @param weights Raw weights bytes; borrowed, keep alive until #IWriter::finalize().
 * @return Encoder for the sized or unsized pointer-mode #IWriter::add_section() overload.
 */
OPENVINO_RUNTIME_API SectionEncoderPtr encode_weights(ov::util::MemoryView weights);

/**
 * @brief Copies the payload of a #weights_tag section into an owned byte buffer.
 * @return The weights bytes, or `std::nullopt` when the section cannot be read.
 */
OPENVINO_RUNTIME_API std::optional<WeightsBuffer> decode_weights(const Section& section) noexcept;

/**
 * @brief Views a pointer-mode #weights_tag section without copying, when backed by addressable memory.
 * @note The buffer backing the section must outlive the returned view.
 * @return The weights view, or `std::nullopt` for an inline or stream-backed section.
 */
OPENVINO_RUNTIME_API std::optional<ov::util::MemoryView> decode_weights_view(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
