// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/matmul_cpu.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "snippets/utils.hpp"
#include "matmul_shape_inference.hpp"

namespace ngraph {
namespace snippets {
namespace op {

MatMulCPU::MatMulCPU(const Output<Node>& A, const Output<Node>& B) : MatMul(), m_output_layout({}) {
    set_arguments({A, B});
    set_output_size(1);
    constructor_validate_and_infer_types();
}

MatMulCPU::MatMulCPU(const Output<Node>& A, const Output<Node>& B, std::vector<size_t> output_layout)
    : MatMul(), m_output_layout(std::move(output_layout)) {
    set_arguments({A, B});
    set_output_size(1);
    constructor_validate_and_infer_types();
}

bool MatMulCPU::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(MatMulCPU_visit_attributes);
    // todo: should we visit planar shapes?
    //visitor.on_attribute("leading_dimensions", m_leading_dimensions);
    return true;
}

void MatMulCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(MatMulCPU_validate_and_infer_types);
    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");

    std::vector<ov::PartialShape> planar_input_shapes;
    for (const auto& in : input_values())
        planar_input_shapes.emplace_back(utils::get_port_planar_shape(in));

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    ov::op::v0::shape_infer(this, planar_input_shapes, output_shapes);
    if (get_output_size() == 1) {
        //  We can check for output rt_info only if the output tensor was actually initialized
        //  the easiest way to check is - is to make sure that the out tensor is static
        const auto& tensor = get_output_descriptor(0).get_tensor_ptr();
        if (tensor && tensor->get_partial_shape().is_static()) {
            std::vector<size_t> layout = utils::get_port_layout(tensor);
            if (!layout.empty()) {
                m_output_layout = std::move(layout);
            } else if (!m_output_layout.empty()) {
                tensor->get_rt_info()["Layout"] = m_output_layout;
            }
        }
    }
    if (!m_output_layout.empty()) {
        auto& out_shape = output_shapes[0];
        std::vector<Dimension> reordered_shape(m_output_layout.size());
        // Note: layout[i] are guaranteed to fall inside original_shape by utils::get_port_layout(in)
        for (int i = 0; i < m_output_layout.size(); i++)
            reordered_shape[i] = out_shape[m_output_layout[i]];
        out_shape = std::move(ov::PartialShape(reordered_shape));
    }
    // todo: here we should handle non-planar output layouts
    set_output_type(0, result_et, output_shapes[0]);

    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "MatMulCPU currently supports only static shapes.");
}

std::shared_ptr<Node> MatMulCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(MatMulCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
//    auto new_matmul = std::make_shared<MatMulCPU>(new_args.at(0), new_args.at(1));
    return std::shared_ptr<Node>(new MatMulCPU(new_args.at(0), new_args.at(1), m_output_layout));
//    new_matmul->output_layout = output_layout;
//    return new_matmul;
//    return std::make_shared<MatMulCPU>(new_args.at(0), new_args.at(1));
}

} // namespace op
} // namespace snippets
} // namespace ngraph
