// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/strides.hpp>
#include <ngraph/node_input.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/ov_visibility.hpp>
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
OPENVINO_API bool has_strides_prop(const ngraph::Input<ngraph::Node>& node);
OPENVINO_API ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node);
OPENVINO_API void insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides);
class OPENVINO_API StridesPropagation : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("strides_propagation", "0");
    StridesPropagation() = default;
    StridesPropagation(const ngraph::Strides& value) : value{value} {}
    ngraph::Strides value;
};
} // namespace ov
