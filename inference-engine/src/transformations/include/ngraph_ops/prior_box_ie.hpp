// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"
#include "ngraph/op/experimental/layers/prior_box.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(PriorBoxIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"PriorBoxIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    /// \brief Constructs a PriorBoxIE operation
    ///
    /// \param layer    Layer for which prior boxes are computed
    /// \param image    Input Input to which prior boxes are scaled
    /// \param attrs          PriorBox attributes
    PriorBoxIE(const Output<Node>& input,
               const Output<Node>& image,
               const ngraph::op::PriorBoxAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const PriorBoxAttrs& get_attrs() const { return m_attrs; }

private:
    PriorBoxAttrs m_attrs;
};

}  // namespace op
}  // namespace ngraph
