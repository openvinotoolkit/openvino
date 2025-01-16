// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/brgemm.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace op {

namespace {
std::vector<size_t> get_output_layout(const std::shared_ptr<const ov::Node>& n) {
    const auto& key = lowered::PortDescriptorVectorAttribute::get_type_info_static();
    auto& rt_info = n->get_rt_info();
    const auto& found = rt_info.find(key);
    if (found != rt_info.end()) {
        const auto& out_descs = found->second.as<lowered::PortDescriptorVectorAttribute>().outputs;
        if (out_descs.size() != n->get_output_size())
            OPENVINO_THROW("Get output port descriptor is failed: incorrect count");
        const auto& port_desc = out_descs[0];
        return port_desc->get_layout();
    }
    return {};
}

} // namespace

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B,
               const size_t offset_a, const size_t offset_b, const size_t offset_c,
               std::vector<size_t> layout_a, std::vector<size_t> layout_b, std::vector<size_t> layout_c)
    : MemoryAccess(std::set<size_t>{0, 1}, std::set<size_t>{0}), Op({A, B}) {
    set_output_size(1);
    set_input_offset(offset_a, 0);
    set_input_offset(offset_b, 1);
    set_output_offset(offset_c, 0);
    custom_constructor_validate_and_infer_types(std::move(layout_a), std::move(layout_b), std::move(layout_c));
}

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B,
               const PortDescriptor& desc_a, const PortDescriptor& desc_b, const PortDescriptor& desc_c,
               std::vector<size_t> layout_a, std::vector<size_t> layout_b, std::vector<size_t> layout_c)
    : MemoryAccess(PortMap{{0, desc_a}, {1, desc_b}}, PortMap{{0, desc_c}}), Op({A, B}) {
    set_output_size(1);
    custom_constructor_validate_and_infer_types(std::move(layout_a), std::move(layout_b), std::move(layout_c));
}

void Brgemm::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_a, std::vector<size_t> layout_b, std::vector<size_t> layout_c) {
    INTERNAL_OP_SCOPE(BrgemmCPU_constructor_validate_and_infer_types);

    // During ctor call, Brgemm doesn't know his port descriptors.
    // So we use explicit layouts from parameters
    const auto planar_input_shapes =
            std::vector<ov::PartialShape>{ ov::snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
                                           ov::snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b) };
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), ov::snippets::utils::get_planar_pshape(output_shape, layout_c));
}

void Brgemm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Brgemm_validate_and_infer_types);

    const auto planar_input_shapes = get_planar_input_shapes(inputs());
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));
}

std::shared_ptr<Node> Brgemm::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Brgemm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Brgemm>(new_args.at(0), new_args.at(1),
                                    get_input_port_descriptor(0), get_input_port_descriptor(1), get_output_port_descriptor(0),
                                    lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
                                    lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
                                    lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

bool Brgemm::visit_attributes(AttributeVisitor& visitor) {
    return MemoryAccess::visit_attributes(visitor);
}

ov::element::Type Brgemm::get_output_type(const ov::element::Type& in_type0, const ov::element::Type& in_type1) {
    const bool is_f32 = utils::everyone_is(element::f32, in_type0, in_type1);
    const bool is_int8 = utils::one_of(in_type0, element::i8, element::u8) && in_type1 == element::i8;
    const bool is_bf16 = utils::everyone_is(element::bf16, in_type0, in_type1);
    if (is_f32 || is_bf16) {
        return element::f32;
    } else if (is_int8) {
        return element::i32;
    } else {
        return element::undefined;
    }
}

ov::element::Type Brgemm::get_output_type() const {
    auto output_type = get_output_type(get_input_element_type(0), get_input_element_type(1));
    if (output_type == element::undefined) {
        OPENVINO_THROW("BrgemmCPU node has incompatible input element types: " +
                       get_input_element_type(0).get_type_name() +
                       " and " +
                       get_input_element_type(1).get_type_name());
    }

    return output_type;
}

std::vector<ov::PartialShape> Brgemm::get_planar_input_shapes(const std::vector<ov::Input<ov::Node>>& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Brgemm::get_planar_input_shapes() expects 2 inputs");
    return { utils::get_planar_pshape(inputs[0]), utils::get_planar_pshape(inputs[1]) };
}

ov::PartialShape Brgemm::get_planar_output_shape(const ov::PartialShape& output_shape) const {
    // This method can be safely called from validate_and_infer_types() before output creation
    const auto& out_layout  = get_output_layout(shared_from_this());
    if (!out_layout.empty())
        return utils::get_planar_pshape(output_shape, out_layout);

    return output_shape;
}

ov::PartialShape Brgemm::infer_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const {
    OPENVINO_ASSERT(input_shapes.size() == 2, "BRGEMM expects 2 input shapes for shape inference");

    // Note: All majors checks are missed because Brgemm is transformed from MatMul with whole shape infer support

    const auto arg0_shape = input_shapes[0];
    const auto arg1_shape = input_shapes[1];

    size_t arg0_rank = arg0_shape.size(), arg1_rank = arg1_shape.size();

    // temporary shapes to calculate output shape
    ov::PartialShape arg0_shape_tmp(arg0_shape), arg1_shape_tmp(arg1_shape);

    // one-dimensional tensors unsqueezing is applied to each input independently.
    if (arg0_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), 1);
        arg0_rank = arg0_shape_tmp.size();
    }
    if (arg1_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg1_shape_tmp.insert(arg1_shape_tmp.end(), 1);
        arg1_rank = arg1_shape_tmp.size();
    }
    // Check matrices dimensions compatibility,
    using DimType = typename std::iterator_traits<typename ov::PartialShape::iterator>::value_type;
    auto merged_dimension = DimType();
    auto arg0_col_dim = arg0_shape_tmp[arg0_rank - 1];
    auto arg1_row_dim = arg1_shape_tmp[arg1_rank - 2];
    OPENVINO_ASSERT(DimType::merge(merged_dimension, arg0_col_dim, arg1_row_dim) || arg0_col_dim.is_dynamic() || arg1_row_dim.is_dynamic(),
                    "Incompatible Brgemm matrix dimension. arg0_col_dim = ", arg0_col_dim, ", arg1_row_dim = ", arg1_row_dim);

    // add 1 to begin to align shape ranks if needed
    if (arg0_rank < arg1_rank)
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    else if (arg0_rank > arg1_rank)
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);

    size_t max_rank = arg0_shape_tmp.size();
    std::vector<DimType> output_shape(max_rank);
    for (size_t i = 0; i < max_rank - 2; ++i) {
         OPENVINO_ASSERT(DimType::broadcast_merge(output_shape[i], arg0_shape_tmp[i], arg1_shape_tmp[i]) ||
                         arg0_shape_tmp[i].is_dynamic() ||
                         arg1_shape_tmp[i].is_dynamic(),
                        "Incompatible Brgemm batch dimension");
    }
    output_shape[output_shape.size() - 2] = arg0_shape_tmp[arg0_shape_tmp.size() - 2];  // M
    output_shape[output_shape.size() - 1] = arg1_shape_tmp[arg1_shape_tmp.size() - 1];  // N

    // removing the temporary axes from originally 1D tensors.
    if (arg0_shape.rank().get_length() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 2);
    }
    if (arg1_shape.rank().get_length() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 1);
    }
    return output_shape;
}
} // namespace op
} // namespace snippets
} // namespace ov
