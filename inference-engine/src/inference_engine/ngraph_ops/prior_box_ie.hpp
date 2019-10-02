// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"
#include "ngraph/op/experimental/layers/prior_box.hpp"

namespace ngraph {
namespace op {
class PriorBoxIE : public Op {
public:
    /// \brief Constructs a PriorBoxIE operation
    ///
    /// \param layer    Layer for which prior boxes are computed
    /// \param image    Input Input to which prior boxes are scaled
    /// \param attrs          PriorBox attributes
    PriorBoxIE(const std::shared_ptr<Node>& input,
               const std::shared_ptr<Node>& image,
               const ngraph::op::PriorBoxAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const PriorBoxAttrs& get_attrs() const { return m_attrs; }

private:
    PriorBoxAttrs m_attrs;
};
}  // namespace op
}  // namespace ngraph
