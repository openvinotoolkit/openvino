// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"

#include "brgemm_copy_b.hpp"

#include "utils/general_utils.h"

using namespace std;
using namespace ov;

intel_cpu::BrgemmCopyBBase::BrgemmCopyBBase(const Output<Node>& x, const element::Type src_type,
                                            const size_t offset_in, const size_t offset_out)
    : ngraph::snippets::op::MemoryAccess({x}), m_src_type(src_type) {
    set_input_port_descriptor({0, offset_in}, 0);
    set_output_port_descriptor({0, offset_out}, 0);
}

void intel_cpu::BrgemmCopyBBase::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCopyBBase_validate_and_infer_types);

    const auto element_type = get_input_element_type(0);
    NGRAPH_CHECK(one_of(element_type, element::bf16, element::i8),
                 "BrgemmCopyBBase doesn't support element type" + element_type.get_type_name());

    const auto pshape = ngraph::snippets::utils::get_port_planar_shape(input_value(0));
    if (pshape.is_dynamic()) {
        set_output_type(0, element_type, ov::PartialShape{ov::Dimension::dynamic()});
        return;
    }

    const auto shape = pshape.get_shape();
    const auto N = *shape.rbegin();
    const auto K = *(shape.rbegin() + 1);
    const auto N_blk = element_type == element::bf16 ? 32 : 64;
    const auto brgemmVNNIFactor = 4 / m_src_type.size();

    set_output_type(0, element_type, ov::PartialShape{ov::Dimension(rnd_up(K, brgemmVNNIFactor)),
                                                      ov::Dimension(rnd_up(N, N_blk))});
}

bool intel_cpu::BrgemmCopyBBase::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmCopyBBase_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("src_type", m_src_type);
    return true;
}

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type,
                                    const size_t offset_in, const size_t offset_out)
    : BrgemmCopyBBase(x, src_type, offset_in, offset_out) {
    set_output_size(1);
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCopyB_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(new_args.at(0), m_src_type,
                                         get_offset_in(),
                                         get_offset_out());
}

intel_cpu::BrgemmCopyBWithCompensations::BrgemmCopyBWithCompensations(const Output<Node>& x, const element::Type src_type,
                                                                      const size_t offset_in, const size_t offset_out0, const size_t offset_out1)
    : BrgemmCopyBBase(x, src_type, offset_in, offset_out0) {
    set_output_port_descriptor({0, offset_out1}, 1);
    set_output_size(2);
    constructor_validate_and_infer_types();
}

void intel_cpu::BrgemmCopyBWithCompensations::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCopyBWithCompensations_validate_and_infer_types);
    BrgemmCopyBBase::validate_and_infer_types();

    const auto pshape = ngraph::snippets::utils::get_port_planar_shape(input_value(0));
    if (pshape.is_dynamic()) {
        set_output_type(1, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic()});
        return;
    }

    const auto shape = pshape.get_shape();
    const auto N = *shape.rbegin();
    const auto N_blk = get_input_element_type(0) == element::bf16 ? 32 : 64;

    set_output_type(1, ov::element::f32, ov::PartialShape{ov::Dimension(rnd_up(N, N_blk))});
}

std::shared_ptr<Node> intel_cpu::BrgemmCopyBWithCompensations::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCopyBWithCompensations_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyBWithCompensations>(new_args.at(0), m_src_type,
                                                          get_offset_in(),
                                                          get_offset_out(),
                                                          get_offset_comp());
}
