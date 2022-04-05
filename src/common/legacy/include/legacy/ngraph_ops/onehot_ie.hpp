// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_api.h>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/one_hot.hpp"

namespace ngraph {
namespace op {

class OneHotIE;

}  // namespace op
}  // namespace ngraph

class ngraph::op::OneHotIE : public Op {
public:
    OPENVINO_OP("OneHotIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    explicit OneHotIE(const Output<ngraph::Node>& input, int axis, int depth, float on_value, float off_value, element::Type type);

    OPENVINO_SUPPRESS_DEPRECATED_START
    size_t get_version() const override { return 1; }
    OPENVINO_SUPPRESS_DEPRECATED_END

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;

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
