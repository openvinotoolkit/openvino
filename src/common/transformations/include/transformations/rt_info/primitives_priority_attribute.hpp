// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines primitives priority attribute
 * @file primitives_priority_attribute.hpp
 */

#pragma once

#include <assert.h>

#include <functional>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <set>
#include <string>

#include "openvino/core/runtime_attribute.hpp"

namespace ov {
/**
 * @ingroup ie_runtime_attr_api
 * @brief getPrimitivesPriority return string with primitive priorities value
 * @param[in] node The node will be used to get PrimitivesPriority attribute
 */
NGRAPH_API std::string getPrimitivesPriority(const std::shared_ptr<ngraph::Node>& node);

class NGRAPH_API PrimitivesPriority : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("primitives_priority", "0");

    PrimitivesPriority() = default;

    PrimitivesPriority(const std::string& value) : value(value) {}

    Any merge(const ngraph::NodeVector& nodes) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::string to_string() const override;

    std::string value;
};
}  // namespace ov
