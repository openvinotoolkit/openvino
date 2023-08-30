// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
#include "snippets/op/buffer.hpp"

#include "brgemm_copy_b.hpp"

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                                    const size_t offset_in, const size_t offset_out0, const size_t offset_out1,
                                    std::vector<size_t> layout_input, const size_t blk_size_k, const size_t blk_size_n)
    : snippets::op::MemoryAccess({x}, 1, type == Type::WithCompensations ? 2 : 1),
      m_type(type), m_src_type(src_type) {
    m_brgemmVNNIFactor = 4 / m_src_type.size();
    set_output_size(type == Type::WithCompensations ? 2 : 1);
    set_input_port_descriptor({0, offset_in}, 0);
    set_output_port_descriptor({0, offset_out0}, 0);
    if (is_with_compensations()) {
        set_output_port_descriptor({0, offset_out1}, 1);
    }
    compute_block_size_values(blk_size_k, blk_size_n);
    custom_constructor_validate_and_infer_types(std::move(layout_input));
}

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                                    const PortDescriptor& desc_in0, const PortDescriptor& desc_out0, const PortDescriptor& desc_out1,
                                    std::vector<size_t> layout_input, const size_t blk_size_k, const size_t blk_size_n)
    : snippets::op::MemoryAccess({x}, 1, type == Type::WithCompensations ? 2 : 1),
      m_type(type), m_src_type(src_type) {
    m_brgemmVNNIFactor = 4 / m_src_type.size();
    set_output_size(type == Type::WithCompensations ? 2 : 1);
    set_input_port_descriptor(desc_in0, 0);
    set_output_port_descriptor(desc_out0, 0);
    if (is_with_compensations()) {
        set_output_port_descriptor(desc_out1, 1);
    }
    compute_block_size_values(blk_size_k, blk_size_n);
    custom_constructor_validate_and_infer_types(std::move(layout_input));
}

bool BrgemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmRepack_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("src_type", m_src_type);
    return true;
}

void BrgemmCopyB::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input) {
    INTERNAL_OP_SCOPE(BrgemmRepack_ctor_validate_and_infer_types);
    // During ctor call, BrgemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto element_type = get_input_element_type(0);
    const auto pshape = snippets::utils::get_reordered_planar_shape(get_input_partial_shape(0), layout_input);
    validate(pshape, element_type);
}

void BrgemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmRepack_validate_and_infer_types);

    const auto element_type = get_input_element_type(0);
    const auto pshape = snippets::utils::get_port_planar_shape(input(0));
    validate(pshape, element_type);
}

void BrgemmCopyB::validate(const ov::PartialShape& pshape, const ov::element::Type& element_type) {
    NGRAPH_CHECK(one_of(element_type, element::bf16, element::i8),
                "BrgemmCopyB doesn't support element type" + element_type.get_type_name());

    if (pshape.is_dynamic()) {
        set_output_type(0, element_type, ov::PartialShape {ov::Dimension::dynamic()});
        if (is_with_compensations()) {
            set_output_type(1, ov::element::f32, ov::PartialShape {ov::Dimension::dynamic()});
        }
        return;
    }

    const auto shape = pshape.get_shape();
    const auto N = *shape.rbegin();
    const auto K = *(shape.rbegin() + 1);

    set_output_type(0, element_type, ov::PartialShape{ov::Dimension(rnd_up(K, m_brgemmVNNIFactor)),
                                                      ov::Dimension(rnd_up(N, m_N_blk))});
    if (is_with_compensations()) {
        set_output_type(1, ov::element::f32, ov::PartialShape{ov::Dimension(rnd_up(N, m_N_blk))});
    }
}

void intel_cpu::BrgemmCopyB::compute_block_size_values(const size_t blk_size_k, const size_t blk_size_n) {
    const auto input_shape = snippets::utils::get_port_planar_shape(input(0)).get_shape();
    m_K_blk = blk_size_k != 0 ? blk_size_k : *(input_shape.rbegin() + 1);
    m_N_blk = blk_size_n != 0 ? blk_size_n : *input_shape.rbegin();
}

std::shared_ptr<Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(new_args.at(0), m_src_type, m_type,
                                         get_input_port_descriptor(0),
                                         get_output_port_descriptor(0),
                                         is_with_compensations() ? get_output_port_descriptor(1) : PortDescriptor{},
                                         snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
                                         m_K_blk, m_N_blk);
}

size_t BrgemmCopyB::get_offset_compensations() const {
    OPENVINO_ASSERT(is_with_compensations() && get_output_size() == 2,
                    "The offset for compensations must be in BrgemmCopyB only with compensations and 2 outputs!");
    return get_output_offset(1);
}

BrgemmCopyB::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& brg_copyb = ov::as_type_ptr<BrgemmCopyB>(n);
    OPENVINO_ASSERT(brg_copyb, "Got invalid node in BrgemmCopyB::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
    m_num_outs = brg_copyb->get_output_size();
    m_N_blk = brg_copyb->get_n_block_size();
    m_brgemmVNNIFactor = brg_copyb->m_brgemmVNNIFactor;
}

snippets::IShapeInferSnippets::Result
BrgemmCopyB::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    const auto& old_shape = input_shapes[0].get();
    IShapeInferSnippets::VectorDims planar_shape;
    planar_shape.reserve(old_shape.size());
    for (const auto idx : m_layout)
        planar_shape.push_back(old_shape[idx]);
    const auto N = *planar_shape.rbegin();
    const auto K = *(planar_shape.rbegin() + 1);
    OPENVINO_ASSERT(N != DYNAMIC_DIMENSION && K != DYNAMIC_DIMENSION,
                    "BrgemmCopyB shape infer got dynamic N or K dimension, which is not supported");

    std::vector<IShapeInferSnippets::VectorDims> new_shapes(m_num_outs);
    new_shapes[0].push_back(rnd_up(K, m_brgemmVNNIFactor));
    new_shapes[0].push_back(rnd_up(N, m_N_blk));
    if (m_num_outs == 2) {
        new_shapes[1].push_back(rnd_up(N, m_N_blk));
    }
    return {new_shapes, snippets::ShapeInferStatus::success};
}

} // namespace intel_cpu
} // namespace ov