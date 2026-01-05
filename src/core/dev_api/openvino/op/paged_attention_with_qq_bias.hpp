// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

class OPENVINO_API paged_attention_with_qq_bias : public ov::op::Op {
public:
    OPENVINO_OP("paged_attention_with_qq_bias");

    paged_attention_with_qq_bias() = default;
    paged_attention_with_qq_bias(const OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void set_out_type(int index, const ov::element::Type& output_type);
protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic};
};

}  // namespace op
}  // namespace ov
