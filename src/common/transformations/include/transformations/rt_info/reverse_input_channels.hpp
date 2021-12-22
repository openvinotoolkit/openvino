// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <openvino/core/runtime_attribute.hpp>

namespace ov {
NGRAPH_API bool is_reverse_input_channels(const std::shared_ptr<ngraph::Node>& node);

NGRAPH_API void set_is_reverse_input_channels(std::shared_ptr<ngraph::Node> node);

/*
 * ReverseInputChannels attribute indicates that operation can be fused
 * by ReverseInputChannels fusion transformation for cases when shape is dynamic.
 */
class NGRAPH_API ReverseInputChannels : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("reverse_input_channels", "0");
    ReverseInputChannels() = default;
    bool visit_attributes(AttributeVisitor& visitor) override { return true; };
};
} // namespace ov
