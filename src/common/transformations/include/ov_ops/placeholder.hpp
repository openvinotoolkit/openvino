// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief GRUSequence operation.
///
/// Supposed to be used in place of an optional input of another operation

class TRANSFORMATIONS_API Placeholder : public ov::op::Op {
public:
    OPENVINO_OP("Placeholder", "ie_internal_opset");

    Placeholder();

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
