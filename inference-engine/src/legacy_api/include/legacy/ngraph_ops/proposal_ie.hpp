// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include <ngraph/op/proposal.hpp>
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(ProposalIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"ProposalIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    //  \brief Constructs a Proposal operation
    //
    //  \param class_probs     Class probability scores
    //  \param class_bbox_deltas    Class prediction bbox_deltas
    //  \param image_shape     Shape of image
    //  \param attrs           Proposal op attributes
    ProposalIE(const Output<Node>& class_probs,
               const Output<Node>& class_bbox_deltas,
               const Output<Node>& image_shape,
               const ProposalAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node>
    clone_with_new_inputs(const OutputVector& new_args) const override;

    const ProposalAttrs& get_attrs() const { return m_attrs; }

private:
    ProposalAttrs m_attrs;
};
}  // namespace op
}  // namespace ngraph
