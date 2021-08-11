// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"
#include "ngraph/op/topk.hpp"

namespace ov {
namespace op {

class INFERENCE_ENGINE_API_CLASS(TopKIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"TopKIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    TopKIE(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const ov::op::TopKMode mode,
           const ov::op::TopKSortType sort,
           const element::Type& index_element_type = element::i32);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() { return m_axis;}

    ov::op::TopKMode get_mode() { return m_mode; }

    ov::op::TopKSortType get_sort_type() { return m_sort_type; }
    bool visit_attributes(AttributeVisitor &visitor) override;

private:
    int64_t m_axis;
    ov::op::TopKMode m_mode;
    ov::op::TopKSortType m_sort_type;
    ov::element::Type m_index_element_type;
};

}  // namespace op
}  // namespace ov
