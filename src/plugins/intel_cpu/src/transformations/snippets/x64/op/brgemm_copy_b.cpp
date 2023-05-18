// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"

#include "brgemm_copy_b.hpp"

#include "utils/general_utils.h"

using namespace std;
using namespace ov;

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                                    const size_t offset_in, const size_t offset_out0, const size_t offset_out1)
    : ngraph::snippets::op::MemoryAccess({x}, 1, type == Type::WithCompensations ? 2 : 1), m_type(type), m_src_type(src_type) {
    set_output_size(get_output_port_count());
    m_input_ports.resize(get_input_size());
    m_output_ports.resize(get_output_size());
    set_input_port_descriptor({0, offset_in}, 0);
    set_output_port_descriptor({0, offset_out0}, 0);
    if (is_with_compensations()) {
        set_output_port_descriptor({0, offset_out1}, 1);
    }
    constructor_validate_and_infer_types();
}

bool intel_cpu::BrgemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmRepack_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("src_type", m_src_type);
    return true;
}

void intel_cpu::BrgemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmRepack_validate_and_infer_types);

    const auto element_type = get_input_element_type(0);
    NGRAPH_CHECK(one_of(element_type, element::bf16, element::i8),
                 "BrgemmCopyB doesn't support element type" + element_type.get_type_name());

    const auto pshape = ngraph::snippets::utils::get_port_planar_shape(input_value(0));
    if (pshape.is_dynamic()) {
        set_output_type(0, element_type, ov::PartialShape{ov::Dimension::dynamic()});
        if (is_with_compensations()) {
            set_output_type(1, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic()});
        }
        return;
    }

    const auto shape = pshape.get_shape();
    const auto N = *shape.rbegin();
    const auto K = *(shape.rbegin() + 1);
    const auto N_blk = element_type == element::bf16 ? 32 : 64;
    const auto brgemmVNNIFactor = 4 / m_src_type.size();

    set_output_type(0, element_type, ov::PartialShape{ov::Dimension(rnd_up(K, brgemmVNNIFactor)),
                                                      ov::Dimension(rnd_up(N, N_blk))});
    if (is_with_compensations()) {
        set_output_type(1, ov::element::f32, ov::PartialShape{ov::Dimension(rnd_up(N, N_blk))});
    }
}

std::shared_ptr<Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(new_args.at(0), m_src_type, m_type,
                                         get_offset_in(),
                                         get_offset_out(),
                                         is_with_compensations() ? get_offset_compensations() : 0);
}

size_t intel_cpu::BrgemmCopyB::get_offset_compensations() const {
    OPENVINO_ASSERT(is_with_compensations() && get_output_size() == 2,
                    "The offset for compensations must be in BrgemmCopyB only with compensations and 2 outputs!");
    return get_output_offset(1);
}
