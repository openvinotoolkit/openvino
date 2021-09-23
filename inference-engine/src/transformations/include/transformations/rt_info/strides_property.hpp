// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/strides.hpp>
#include <ngraph/node_input.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

namespace ov {
template <>
class TRANSFORMATIONS_API VariantWrapper<ngraph::Strides> : public VariantImpl<ngraph::Strides> {
public:
    OPENVINO_RTTI("strides", "0");

    VariantWrapper() = default;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

TRANSFORMATIONS_API bool has_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API void insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides);
} // namespace ov
