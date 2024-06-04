// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/gather.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API GatherCompressed : public ov::op::v8::Gather {
public:
    OPENVINO_OP("GatherCompressed", "ie_internal_opset");

    GatherCompressed() = default;

    GatherCompressed(const ov::Output<Node>& data,
                     const ov::Output<Node>& indices,
                     const ov::Output<Node>& axis,
                     const int64_t batch_dims,
                     const ov::Output<Node>& decompression_scale,
                     const ov::Output<Node>& decompression_zero_point);

    GatherCompressed(const ov::Output<Node>& data,
                     const ov::Output<Node>& indices,
                     const ov::Output<Node>& axis,
                     const int64_t batch_dims,
                     const ov::Output<Node>& decompression_scale);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override {
        return false;
    }
};

}  // namespace internal
}  // namespace op
}  // namespace ov
