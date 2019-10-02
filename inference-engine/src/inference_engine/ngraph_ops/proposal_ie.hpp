//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>

#include <ngraph/op/experimental/layers/proposal.hpp>
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class ProposalIE : public Op {
public:
    //  \brief Constructs a Proposal operation
    //
    //  \param class_probs     Class probability scores
    //  \param class_logits    Class prediction logits
    //  \param image_shape     Shape of image
    //  \param attrs           Proposal op attributes
    ProposalIE(const std::shared_ptr<Node>& class_probs,
               const std::shared_ptr<Node>& class_logits,
               const std::shared_ptr<Node>& image_shape,
               const ProposalAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node>
    copy_with_new_args(const NodeVector& new_args) const override;

    const ProposalAttrs& get_attrs() const { return m_attrs; }

private:
    ProposalAttrs m_attrs;
};
}  // namespace op
}  // namespace ngraph
