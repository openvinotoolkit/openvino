// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node_input.hpp>
#include <ngraph/strides.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

#include "openvino/core/runtime_attribute.hpp"

namespace ov {
TRANSFORMATIONS_API bool has_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API void insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides);
class TRANSFORMATIONS_API StridesPropagation : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("strides_propagation", "0");
    StridesPropagation() = default;
    StridesPropagation(const ngraph::Strides& value) : value{value} {}
    ngraph::Strides value;
};
}  // namespace ov
