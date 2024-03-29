// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "matmul_shape_inference.hpp"
#include "broadcast_shape_inference.hpp"
#include "reshape_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

Gemm::Gemm(const ov::Output<Node>& A,
           const ov::Output<Node>& B,
           const std::vector<int64_t>& order_a,
           const std::vector<int64_t>& order_b,
           const std::vector<int64_t>& order_c,
           const ov::element::Type output_type)
    : ov::op::v0::MatMul()
    , m_pattern_a({})
    , m_pattern_b({})
    , m_order_a(order_a)
    , m_order_b(order_b)
    , m_order_c(order_c)
    , m_output_type(output_type) {
    set_arguments({A, B});
    set_transpose_a(false);
    set_transpose_b(false);
    validate_and_infer_types();
}

Gemm::Gemm(const ov::Output<Node>& A,
           const ov::Output<Node>& B,
           const std::vector<int64_t>& pattern_a,
           const std::vector<int64_t>& pattern_b,
           const std::vector<int64_t>& order_a,
           const std::vector<int64_t>& order_b,
           const std::vector<int64_t>& order_c,
           const ov::element::Type output_type)
    : ov::op::v0::MatMul()
    , m_pattern_a(pattern_a)
    , m_pattern_b(pattern_b)
    , m_order_a(order_a)
    , m_order_b(order_b)
    , m_order_c(order_c)
    , m_output_type(output_type) {
    set_arguments({A, B});
    set_transpose_a(false);
    set_transpose_b(false);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> Gemm::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<Gemm>(new_args.at(0), new_args.at(1), m_pattern_a, m_pattern_b, m_order_a, m_order_b, m_order_c, m_output_type);
}

void Gemm::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 2,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 2.");

    auto out_shapes = shape_infer(this,
                                  std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)},
                                  m_pattern_a,
                                  m_pattern_b,
                                  m_order_a,
                                  m_order_b,
                                  m_order_c);

    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool Gemm::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("pattern_a", m_pattern_a);
    visitor.on_attribute("pattern_b", m_pattern_b);
    visitor.on_attribute("order_a", m_order_a);
    visitor.on_attribute("order_b", m_order_b);
    visitor.on_attribute("order_c", m_order_c);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::vector<ov::PartialShape> shape_infer(const Gemm* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& pattern_a,
                                          const std::vector<int64_t>& pattern_b,
                                          const std::vector<int64_t>& order_a,
                                          const std::vector<int64_t>& order_b,
                                          const std::vector<int64_t>& order_c) {
    // broadcasted and reshaped shape
    auto get_broadcast_and_reshape_shape = [&](const ov::PartialShape shape, const std::vector<int64_t>& reshape_pattern) {
        ov::PartialShape out_shape;
        if (reshape_pattern.size() > 0) {
            std::vector<ov::Dimension> dims(shape);
            auto reshape_axes = ov::intel_gpu::find_non_val_pos(reshape_pattern, 0);
            if (reshape_axes != reshape_pattern.size()) {
                dims[reshape_axes] = reshape_pattern[reshape_axes];
                out_shape = ov::PartialShape(dims);
            } else {
                out_shape = shape;
            }
        } else {
            out_shape = shape;
        }
        return out_shape;
    };
    auto shape_a = get_broadcast_and_reshape_shape(input_shapes[0], pattern_a);
    auto shape_b = get_broadcast_and_reshape_shape(input_shapes[1], pattern_b);

    // transposed shape
    auto transpose_shape = [](const ov::PartialShape shape, const std::vector<int64_t>& order) {
        auto shape_transposed = ov::PartialShape::dynamic(shape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            shape_transposed[i] = shape[order[i]];
        }

        return shape_transposed;
    };
    auto shape_a_t = (order_a.size() > 1) ? transpose_shape(shape_a, order_a) : shape_a;
    auto shape_b_t = (order_b.size() > 1) ? transpose_shape(shape_b, order_b) : shape_b;
    OPENVINO_ASSERT(op != nullptr, "op should not be nullptr for shape_infer.");
    auto out_shapes = ov::op::v0::shape_infer(dynamic_cast<const ov::op::v0::MatMul*>(op), std::vector<ov::PartialShape>{shape_a_t, shape_b_t});

    if (order_c.size() > 0) {
        return { transpose_shape(out_shapes[0], order_c) };
    } else {
        return { out_shapes[0] };
    }
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
