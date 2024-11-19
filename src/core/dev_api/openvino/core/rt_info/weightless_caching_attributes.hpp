// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
class OPENVINO_API WeightlessCacheAttribute : public RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightlessCacheAttribute");

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype),
          curr_dtype(original_dtype) {}

    bool is_copyable() const override;
    bool is_copyable(const std::shared_ptr<Node>& from, const std::shared_ptr<Node>& to) const override;

    size_t original_size;
    size_t bin_offset;
    ov::element::Type original_dtype;
    ov::element::Type curr_dtype;
};

}  // namespace ov
