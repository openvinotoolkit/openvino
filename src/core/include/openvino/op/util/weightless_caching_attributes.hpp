// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

/**
 * @brief ConstantOriginalSize class represents runtime info attribute that marks
 * the original size of the constant in bytes. It is used in the weightless caching
 * mechanism. It's not copyable in case the data was changed (the original node was
 * replaced by a new one produced during the tranformation pipeline) - in that case
 * weightless caching can't be used for that constant.
 */
class OPENVINO_API ConstantOriginalSize : public RuntimeAttribute {
public:
    OPENVINO_RTTI("ConstantOriginalSize");

    ConstantOriginalSize() = default;

    ConstantOriginalSize(size_t original_size) : original_size(original_size) {}

    bool is_copyable() const override {
        return false;
    }

    size_t original_size;
};

/**
 * @brief ConstantBinOffset class represents runtime info attribute that marks
 * the binary offset of the constant's data in the weights file used by the
 * weightless caching mechanism. It's not copyable in case the data
 * was changed (the original node was replaced by a new one produced during the
 * tranformation pipeline) - in that case weightless caching can't be used for
 * that constant.
 */
class OPENVINO_API ConstantBinOffset : public RuntimeAttribute {
public:
    OPENVINO_RTTI("ConstantBinOffset");

    ConstantBinOffset() = default;

    ConstantBinOffset(size_t bin_offset) : bin_offset(bin_offset) {}

    bool is_copyable() const override {
        return false;
    }

    size_t bin_offset;
};

}  // namespace ov
