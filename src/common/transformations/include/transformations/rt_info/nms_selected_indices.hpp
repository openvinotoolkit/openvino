// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <functional>
#include <memory>
#include <set>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API bool has_nms_selected_indices(const Node* node);

TRANSFORMATIONS_API void set_nms_selected_indices(Node* node);

class TRANSFORMATIONS_API NmsSelectedIndices : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("nms_selected_indices", "0", ov::RuntimeAttribute);
    NmsSelectedIndices() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
