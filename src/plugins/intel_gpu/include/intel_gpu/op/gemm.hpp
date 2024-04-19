// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class Gemm : public ov::op::v0::MatMul {
public:
    OPENVINO_OP("Gemm", "gpu_opset");

    Gemm() = default;

    Gemm(const ov::Output<Node>& A,
         const ov::Output<Node>& B,
         const std::vector<int64_t>& order_a,
         const std::vector<int64_t>& order_b,
         const std::vector<int64_t>& order_c,
         const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::vector<int64_t> get_input0_transpose_order() const { return m_order_a; }
    std::vector<int64_t> get_input1_transpose_order() const { return m_order_b; }
    std::vector<int64_t> get_output_transpose_order() const { return m_order_c; }
    ov::element::Type get_output_type() const { return m_output_type; }

    static std::vector<int64_t> default_order(size_t rank) {
        std::vector<int64_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

protected:
    std::vector<int64_t> m_order_a;
    std::vector<int64_t> m_order_b;
    std::vector<int64_t> m_order_c;
    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const Gemm* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_a,
                                          const std::vector<int64_t>& order_b,
                                          const std::vector<int64_t>& order_c);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
