// Copyright (C) 2018-2020 Intel Corporation
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

class INFERENCE_ENGINE_API_CLASS(TopKIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"TopKIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    TopKIE(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const ngraph::op::TopKMode mode,
           const ngraph::op::TopKSortType sort);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() { return m_axis;}

    ngraph::op::TopKMode get_mode() { return m_mode; }

    ngraph::op::TopKSortType get_sort_type() { return m_sort_type; }

private:
    int64_t m_axis;
    ngraph::op::TopKMode m_mode;
    ngraph::op::TopKSortType m_sort_type;
};

}  // namespace op
}  // namespace ngraph
