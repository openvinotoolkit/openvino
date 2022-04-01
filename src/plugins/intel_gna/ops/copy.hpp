// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ov {
namespace intel_gna {
namespace op {
/// \brief GNA specific copy layer operation
///
class Copy : public ngraph::op::Op {
public:
    OPENVINO_OP("Copy", "intel_gna", ov::op::Op);

    Copy() = default;
    /// \brief Constructs an Copy operation.
    ///
    /// \param [in] arg Input tensor
    Copy(const ngraph::Output<ngraph::Node>& arg);

    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};
} // namespace op
} // namespace intel_gna
} // namespace ov
