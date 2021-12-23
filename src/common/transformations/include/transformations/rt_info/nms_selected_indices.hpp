// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/visibility.hpp>
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

OPENVINO_API bool has_nms_selected_indices(const Node * node);

OPENVINO_API void set_nms_selected_indices(Node * node);

class OPENVINO_API NmsSelectedIndices : ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("nms_selected_indices", "0");
    NmsSelectedIndices() = default;
    bool is_copyable() const override { return false; }
};

}  // namespace ov
