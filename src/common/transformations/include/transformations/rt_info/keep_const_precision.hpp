// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void enable_keep_const_precision(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void disable_keep_const_precision(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_keep_const_precision(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief KeepConstPrecision class represents runtime info attribute that marks a Constant
 * as prohibitted to fuse precision in ConvertPrecision
 */
class TRANSFORMATIONS_API KeepConstPrecision : public RuntimeAttribute {
public:
    OPENVINO_RTTI("keep_const_precision", "0", RuntimeAttribute);

    KeepConstPrecision() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
