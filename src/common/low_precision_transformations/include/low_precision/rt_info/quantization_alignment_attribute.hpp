// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <openvino/core/visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "shared_value_attribute.hpp"
#include "attribute_parameters.hpp"

namespace ngraph {
class OPENVINO_API QuantizationAlignmentAttribute : public SharedAttribute<bool> {
public:
    OPENVINO_RTTI("LowPrecision::QuantizationAlignment", "", ov::RuntimeAttribute, 0);
    QuantizationAlignmentAttribute(const bool value = false);

    static ov::Any create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    void merge(std::vector<ov::Any>& attributes);
    std::string to_string() const override;
};

} // namespace ngraph
