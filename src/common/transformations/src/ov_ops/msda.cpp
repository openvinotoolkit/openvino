// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/msda.hpp"

#include "itt.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "scaled_dot_product_attention_shape_inference.hpp"

#include "ov_ops/augru_sequence.hpp"

#include "augru_sequence_shape_inference.hpp"
#include "itt.hpp"
namespace ov {
namespace op {
namespace internal {

namespace {
// Overload << operator for vectors
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
};  // namespace

MSDA::MSDA(const OutputVector& inputs)
    : Op(inputs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MSDA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MSDA_clone_with_new_inputs);
    return std::make_shared<MSDA>(new_args);
}

bool MSDA::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MSDA_visit_attributes);
    return true;
}

// wzx todo
void MSDA::validate_and_infer_types() {
    // INTERNAL_OP_SCOPE(internal_MSDA_validate_and_infer_types);
    // OPENVINO_ASSERT(get_input_size() == 4, "MSDA must have 4 inputs whereas it has ", get_input_size());

    // auto out_type = get_input_element_type(0);

    // const auto& cu_seqlens_type = get_input_element_type(3);
    // NODE_VALIDATION_CHECK(
    //     this,
    //     cu_seqlens_type.is_integral() || cu_seqlens_type.is_dynamic(),
    //     "The element type of cu_seqlens must be integral.");

    // for (size_t i = 1; i < 3; i++) {
    //     const auto& element_type = get_input_element_type(i);
    //     NODE_VALIDATION_CHECK(this,
    //                           element::Type::merge(out_type, out_type, element_type),
    //                           "Mixed input types of K/V are not supported.");
    // }
    // NODE_VALIDATION_CHECK(this,
    //                       out_type.is_real() || out_type.is_dynamic(),
    //                       "The element type of the input tensor must be a floating-point.");

    // const auto& input_shapes = ov::util::get_node_input_partial_shapes(*this);
    // // const auto output_shapes = shape_infer(this, input_shapes);
    // // transpose shape into BHLS(4D), or HLS(3D)
    // auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
    //     if (order.empty())
    //         return pshape;

    //     auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
    //     for (size_t i = 0; i < order.size(); i++) {
    //         transposed_pshape[i] = pshape[order[i]];
    //     }
    //     return transposed_pshape;
    // };
    // const auto& output_shape = transpose_pshape(input_shapes[0], m_order_q);
    // // std::cout << "----------------- MSDA::validate_and_infer_types() -----------------" << std::endl;
    // // std::cout << "----------------- m_order_q: " << m_order_q <<
    // // "," << "m_order_out: " << m_order_out <<
    // // "," << input_shapes[0] << "->" << output_shape<< std::endl;
    // if (m_order_out.size() > 0) {
    //     set_output_type(0, out_type, transpose_pshape(output_shape, m_order_out));
    // } else {
    //     set_output_type(0, out_type, output_shape);
    // }
}

}  // namespace internal
}  // namespace op
}  // namespace ov