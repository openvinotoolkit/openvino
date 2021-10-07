// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Slice operation.
///
class OPENVINO_API Slice : public Op {
public:
    OPENVINO_OP("Slice", "opset8");

    Slice() = default;

    ///
    /// \brief    Constructs Slice operation.
    ///
    Slice(const Output<Node>& data, const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step);
    Slice(const Output<Node>& data,
          const Output<Node>& start,
          const Output<Node>& stop,
          const Output<Node>& step,
          const Output<Node>& axes);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override;
    bool evaluate(const HostTensorVector&, const HostTensorVector&) const override;

    PartialShape calculate_output_shape(const std::vector<int64_t>& starts,
                                        const std::vector<int64_t>& stops,
                                        const std::vector<int64_t>& steps,
                                        const std::vector<int64_t>& axes,
                                        const PartialShape& data_shape) const;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
