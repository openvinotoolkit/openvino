// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API bool has_strides_prop(const Input<Node>& node);
TRANSFORMATIONS_API ov::Strides get_strides_prop(const Input<Node>& node);
TRANSFORMATIONS_API void insert_strides_prop(Input<Node>& node, const Strides& strides);
TRANSFORMATIONS_API void remove_strides_prop(Input<Node>& node);

class TRANSFORMATIONS_API StridesPropagation : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("strides_propagation", "0", ov::RuntimeAttribute);
    StridesPropagation() = default;
    StridesPropagation(const ov::Strides& value) : value{value} {}

    bool is_copyable() const override {
        return false;
    }

    ov::Strides value;
};
}  // namespace ov
