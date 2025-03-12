// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API Convolution : public ov::op::util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("Convolution", "ie_internal_opset", ov::op::util::ConvolutionFwdPropBase);

    Convolution() = default;

    Convolution(const Output<Node>& data_batch,
                const Output<Node>& filters,
                const Strides& strides,
                const CoordinateDiff& pads_begin,
                const CoordinateDiff& pads_end,
                const Strides& dilations,
                const PadType& auto_pad = PadType::EXPLICIT);

    Convolution(const Output<Node>& data_batch,
                const Output<Node>& filters,
                const Output<Node>& bias,
                const Strides& strides,
                const CoordinateDiff& pads_begin,
                const CoordinateDiff& pads_end,
                const Strides& dilations,
                const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
