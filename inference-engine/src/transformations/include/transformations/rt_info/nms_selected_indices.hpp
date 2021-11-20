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
#include <transformations_visibility.hpp>

namespace ov {

TRANSFORMATIONS_API bool has_nms_selected_indices(const Node * node);

TRANSFORMATIONS_API void set_nms_selected_indices(Node * node);

class TRANSFORMATIONS_API NmsSelectedIndices : public VariantImpl<bool> {
public:
    OPENVINO_RTTI("nms_selected_indices", "0");

    NmsSelectedIndices() = default;

    NmsSelectedIndices(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }
};

}  // namespace ov
