// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

///
/// \brief FusedMHA_RPE operation.
///
/// \ingroup ov_ops_cpp_api
class TRANSFORMATIONS_API FusedMHA_RPE : public ov::op::Op {
public:
    OPENVINO_OP("FusedMHA_RPE", "ie_internal_opset", op::Op);

    FusedMHA_RPE() = default;
    FusedMHA_RPE(const Output<Node>& data,
                 const Output<Node>& sin,
                 const Output<Node>& cos,
                 const Output<Node>& prev_keys,
                 const Output<Node>& prev_values,
                 const Output<Node>& boolean_mask,
                 const Output<Node>& attention_mask,
                 const Output<Node>& keys_multiplier,
                 const size_t& d_head);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    size_t d_head{};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
