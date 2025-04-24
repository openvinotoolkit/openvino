// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void disable_divide_conversion(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_divide_conversion(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool divide_is_nonconvertible(const std::shared_ptr<Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief NonconvertibleDivide class represents runtime info attribute that marks
 * a Divide as prohibitted to transform it to power.
 */
class TRANSFORMATIONS_API NonconvertibleDivide : public RuntimeAttribute {
public:
    OPENVINO_RTTI("nonconvertable_divide", "0", RuntimeAttribute);

    NonconvertibleDivide() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
