// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/** @brief Produces the raw IR XML bytes for a #model_struct_tag section at finalize time. */
OPENVINO_RUNTIME_API SectionEncoderPtr encode_model_struct(std::string xml);

/** @brief Copies a #model_struct_tag section into an XML string, or returns nullopt if it cannot be read. */
OPENVINO_RUNTIME_API std::optional<std::string> decode_model_struct(const Section& section) noexcept;

/**
 * @brief Views the XML bytes of an addressable pointer-mode #model_struct_tag section without copying.
 * @note The buffer backing the section must outlive the returned view.
 * @return The XML view, or nullopt for an inline or stream-backed section.
 */
OPENVINO_RUNTIME_API std::optional<std::string_view> decode_model_struct_view(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
