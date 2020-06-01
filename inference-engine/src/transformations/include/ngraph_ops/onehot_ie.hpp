// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <transformations_visibility.hpp>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/one_hot.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API OneHotIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"OneHotIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    explicit OneHotIE(const Output<ngraph::Node>& input, int axis, int depth, float on_value, float off_value, element::Type type);

    size_t get_version() const override { return 1; }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    int get_axis() { return m_axis; }
    int get_depth() { return m_depth; }
    float get_on_value() { return m_on_value; }
    float get_off_value() { return m_off_value; }

private:
    element::Type m_type;
    int m_axis;
    int m_depth;
    float m_off_value = 0.0;
    float m_on_value = 0.0;
};
}  // namespace op
}  // namespace ngraph
