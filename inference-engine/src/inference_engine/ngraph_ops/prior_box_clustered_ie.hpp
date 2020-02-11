// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/op/op.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>

namespace ngraph {
namespace op {

class PriorBoxClusteredIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"PriorBoxClusteredIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    /// \brief Constructs a PriorBoxClusteredIE operation
    ///
    /// \param layer    Layer for which prior boxes are computed
    /// \param image    Input Input to which prior boxes are scaled
    /// \param attrs          PriorBoxClustered attributes
    PriorBoxClusteredIE(const std::shared_ptr<Node>& input,
               const std::shared_ptr<Node>& image,
               const ngraph::op::PriorBoxClusteredAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const PriorBoxClusteredAttrs& get_attrs() const { return m_attrs; }

private:
    PriorBoxClusteredAttrs m_attrs;
};

}  // namespace op
}  // namespace ngraph

