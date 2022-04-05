// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <functional>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <set>
#include <string>
#include <transformations_visibility.hpp>

#include "openvino/core/runtime_attribute.hpp"

namespace ov {

TRANSFORMATIONS_API bool has_nms_selected_indices(const Node* node);

TRANSFORMATIONS_API void set_nms_selected_indices(Node* node);

class TRANSFORMATIONS_API NmsSelectedIndices : ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("nms_selected_indices", "0");
    NmsSelectedIndices() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
