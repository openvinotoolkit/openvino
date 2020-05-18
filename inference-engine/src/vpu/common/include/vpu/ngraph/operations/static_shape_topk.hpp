// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeTopK : public ngraph::op::Op {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeTopK", 0};

    const NodeTypeInfo& get_type_info() const override { return type_info; }

    using SortType = ngraph::op::TopKSortType;
    using Mode = ngraph::op::TopKMode;

    StaticShapeTopK(const Output<Node>& data,
                    const Output<Node>& k,
                    const int64_t axis,
                    const std::string& mode,
                    const std::string& sort,
                    const element::Type& index_element_type = element::i32);

    StaticShapeTopK(const Output<Node>& data,
                    const Output<Node>& k,
                    const int64_t axis,
                    const Mode mode,
                    const SortType sort,
                    const element::Type& index_element_type = element::i32);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_version() const override { return 1; }

    uint64_t get_axis() const;
    int64_t get_provided_axis() const { return m_axis; }
    void set_axis(const int64_t axis);
    Mode get_mode() const { return m_mode; }
    void set_mode(const Mode mode) { m_mode = mode; }
    SortType get_sort_type() const { return m_sort; }
    void set_sort_type(const SortType sort) { m_sort = sort; }
    element::Type get_index_element_type() const { return m_index_element_type; }
    void set_index_element_type(const element::Type& index_element_type) {
        m_index_element_type = index_element_type;
    }
    size_t get_default_output_index() const override { return no_default_index(); }

protected:
    int64_t m_axis;
    uint64_t m_normalized_axis;
    Mode m_mode;
    SortType m_sort;
    element::Type m_index_element_type{element::i32};

    void generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas) override {
            throw ngraph_error("Forward-propagation-only operation");
    }
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
