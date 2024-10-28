// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class IndirectSDPA : public ov::intel_gpu::op::SDPA {
public:
    OPENVINO_OP("IndirectSDPA", "gpu_opset");

    IndirectSDPA() = default;

    IndirectSDPA(const OutputVector& data_inputs,
                 const ov::Output<Node>& beam_table,
                 const bool is_causal,
                 const int64_t indirect_axis,
                 const std::vector<int64_t>& order_q,
                 const std::vector<int64_t>& order_k,
                 const std::vector<int64_t>& order_v,
                 const std::vector<int64_t>& order_out,
                 const ov::element::Type output_type = ov::element::undefined);

    IndirectSDPA(const OutputVector& data_inputs,
                 const ov::Output<Node>& beam_table,
                 const bool is_causal,
                 const int64_t indirect_axis,
                 const std::vector<int64_t>& order_q,
                 const std::vector<int64_t>& order_k,
                 const std::vector<int64_t>& order_v,
                 const std::vector<int64_t>& order_out,
                 const QuantizationAttribute& quantization_attribute,
                 const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

    int64_t get_indirect_axis() const { return m_indirect_axis; }

    using ov::intel_gpu::op::SDPA::default_order;

protected:
    int64_t m_indirect_axis = -1;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
