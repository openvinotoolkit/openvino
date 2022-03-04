// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"
#include "ngraph/op/topk.hpp"

namespace ngraph {
namespace op {

class TopKIE : public Op {
public:
    OPENVINO_OP("TopKIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    TopKIE(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const ngraph::op::TopKMode mode,
           const ngraph::op::TopKSortType sort,
           const element::Type& index_element_type = element::i32);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() { return m_axis;}

    ngraph::op::TopKMode get_mode() { return m_mode; }

    ngraph::op::TopKSortType get_sort_type() { return m_sort_type; }
    bool visit_attributes(AttributeVisitor &visitor) override;

private:
    int64_t m_axis;
    ngraph::op::TopKMode m_mode;
    ngraph::op::TopKSortType m_sort_type;
    ngraph::element::Type m_index_element_type;
};

}  // namespace op
}  // namespace ngraph
