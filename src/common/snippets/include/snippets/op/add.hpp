// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace snippets {
namespace op {

class Add : public ngraph::opset1::Add {
public:
    OPENVINO_OP("Add", "SnippetsOpset");

    Add(const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY));
    Add(const ngraph::opset1::Add& add);
    Add() = default;

    void validate_and_infer_types() override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
};

} // namespace op
} // namespace snippets
} // namespace ngraph
