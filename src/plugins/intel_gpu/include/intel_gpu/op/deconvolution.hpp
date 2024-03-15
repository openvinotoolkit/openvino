// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/convolution_backprop_base.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

// Common node for v1::Deconvolution and v1::GroupDeconvolution
class Deconvolution : public ov::op::util::ConvolutionBackPropBase {
public:
    OPENVINO_OP("Deconvolution", "gpu_opset", ov::op::util::ConvolutionBackPropBase);

    Deconvolution() = default;

    Deconvolution(const ov::Output<Node>& data_batch,
                const ov::Output<Node>& filters,
                const ov::Output<Node>& bias,
                const ov::Strides& strides,
                const ov::CoordinateDiff& pads_begin,
                const ov::CoordinateDiff& pads_end,
                const ov::Strides& dilations,
                const int64_t& groups,
                const ov::op::PadType& auto_pad,
                const ov::element::Type& output_type,
                const ov::CoordinateDiff& output_padding);

    Deconvolution(const ov::Output<Node>& data_batch,
                const ov::Output<Node>& filters,
                const ov::Output<Node>& bias,
                const ov::Output<Node>& output_shape,
                const ov::Strides& strides,
                const ov::CoordinateDiff& pads_begin,
                const ov::CoordinateDiff& pads_end,
                const ov::Strides& dilations,
                const int64_t& groups,
                const ov::op::PadType& auto_pad,
                const ov::element::Type& output_type,
                const ov::CoordinateDiff& output_padding);


    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_groups() const;
    int64_t get_groups() const;

    bool is_asymmetric() const;

    struct Args {
        static constexpr const size_t INPUT = 0;
        static constexpr const size_t WEIGHTS = 1;
        static constexpr const size_t BIAS = 2;
        static constexpr const size_t OUTPUT_SHAPE = 3;
    };

protected:
    int64_t m_groups = -1; // negative value means no groups
    bool m_asymmetric = false;
    ov::element::Type m_output_type = ov::element::undefined;
};

std::vector<ov::PartialShape> shape_infer(const Deconvolution* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          CoordinateDiff& pads_begin,
                                          CoordinateDiff& pads_end);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
