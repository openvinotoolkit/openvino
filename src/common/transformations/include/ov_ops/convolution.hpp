// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {

class TRANSFORMATIONS_API Convolution : public ov::op::util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("Convolution", "ie_internal_opset", ov::op::util::ConvolutionFwdPropBase);

    Convolution() = default;

    Convolution(const Output<Node>& data_batch,
                const Output<Node>& filters,
                const Output<Node>& bias,
                const Strides& strides,
                const CoordinateDiff& pads_begin,
                const CoordinateDiff& pads_end,
                const Strides& dilations,
                const int64_t groups,
                const PadType& auto_pad,
                const element::Type& output_type);

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
        static constexpr const size_t AZP = 3;
        static constexpr const size_t WZP = 4;
        static constexpr const size_t COMPENSATION = 5;
    };

private:
    int64_t m_groups = -1;  // negative value means no groups
    bool m_asymmetric = false;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
