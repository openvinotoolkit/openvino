// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/gemm.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

class IndirectGemm : public ov::intel_gpu::op::Gemm {
public:
    OPENVINO_OP("IndirectGemm", "gpu_opset", ov::intel_gpu::op::Gemm);

    IndirectGemm() = default;

    IndirectGemm(const ov::Output<Node>& A,
                 const ov::Output<Node>& B,
                 const ov::Output<Node>& I,
                 bool indirect_a,
                 bool indirect_b,
                 int64_t indirect_axis,
                 const std::vector<int64_t>& order_a,
                 const std::vector<int64_t>& order_b,
                 const std::vector<int64_t>& order_c,
                 const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

    bool get_indirect_a() const { return m_indirect_a; }
    bool get_indirect_b() const { return m_indirect_b; }
    int64_t get_indirect_axis() const { return m_indirect_axis; }

    using ov::intel_gpu::op::Gemm::default_order;

protected:
    bool m_indirect_a = false;
    bool m_indirect_b = false;
    int64_t m_indirect_axis = 0;
};

}   // namespace ov::intel_gpu::op
