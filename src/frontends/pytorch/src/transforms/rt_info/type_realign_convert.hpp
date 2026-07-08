// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

/**
 * @brief Marks a Convert/ConvertLike synthesized by an op translator (e.g. translate_linear_ext)
 * that only realigns an fp32 computation back to the original, lower precision, framework dtype.
 * Consumed by RemoveOutputRealignConvert, which removes such converts when they feed only Result
 * node(s) and always erases the marker afterwards, so it never leaks into the exported IR.
 * Copyable so it survives node replacements (e.g. ConvertConvertLike) happening before the pass
 * that consumes it runs.
 */
class TypeRealignConvert : public RuntimeAttribute {
public:
    OPENVINO_RTTI("pt_type_realign_convert", "0", RuntimeAttribute);

    TypeRealignConvert() = default;

    bool is_copyable() const override {
        return true;
    }
};

// Mark `node` (expected to be a Convert/ConvertLike) as a candidate for output-realignment removal.
void mark_type_realign_convert(const std::shared_ptr<Node>& node);

// Remove the marker from `node`, if present. Safe to call unconditionally.
void unmark_type_realign_convert(const std::shared_ptr<Node>& node);

// Return true if `node` currently carries the marker.
bool is_type_realign_convert(const std::shared_ptr<const Node>& node);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
