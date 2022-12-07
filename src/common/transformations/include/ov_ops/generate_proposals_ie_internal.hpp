// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <transformations_visibility.hpp>

#include "openvino/op/generate_proposals.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API GenerateProposalsIEInternal : public op::v9::GenerateProposals {
    using Base = op::v9::GenerateProposals;

public:
    OPENVINO_OP("GenerateProposalsIEInternal", "ie_internal_opset");

    GenerateProposalsIEInternal() = default;

    GenerateProposalsIEInternal(const Output<Node>& im_info,
                                const Output<Node>& anchors,
                                const Output<Node>& deltas,
                                const Output<Node>& scores,
                                const Attributes& attrs,
                                const element::Type& roi_num_type = element::i64);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ngraph {
namespace op {
namespace internal {
using ov::op::internal::GenerateProposalsIEInternal;
}  // namespace internal
}  // namespace op
}  // namespace ngraph
