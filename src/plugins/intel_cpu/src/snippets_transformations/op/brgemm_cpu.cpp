// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "brgemm_cpu.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/utils.hpp"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Type type,
                     const size_t offset_a, const size_t offset_b, const size_t offset_c)
    : Brgemm(), m_type(type) {
    // We call default ctor of Brgemm class to avoid incorrect shape infer in constructor_validate_and_type_infer() call
    set_arguments({A, B});
    set_output_size(1);
    m_input_ports.resize(get_input_size());
    m_output_ports.resize(get_output_size());
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    constructor_validate_and_infer_types();
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch, const Type type,
                     const size_t offset_a, const size_t offset_b, const size_t offset_scratch, const size_t offset_c)
    : Brgemm(), m_type(type) {
    set_arguments({A, B, scratch});
    set_output_size(1);
    m_input_ports.resize(get_input_size());
    m_output_ports.resize(get_output_size());
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    set_input_port_descriptor({0, offset_scratch}, 2);
    constructor_validate_and_infer_types();
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "BrgemmCPU currently supports only static shapes.");

    OPENVINO_ASSERT(implication(one_of(m_type, Type::Floating, Type::WithDataRepacking), get_input_size() == 2),
                    "BrgemmCPU expects 2 inputs in cases, when input precisions are f32|f32, u8|i8 or bf16|bf16 (non-AMX system)");
    OPENVINO_ASSERT(implication(one_of(m_type, Type::WithCompensations, Type::AMX), get_input_size() == 3),
                    "BrgemmCPU expects 3 inputs with input precisions i8|i8 and bf16|bf16 on AMX system");

    const auto brgemm_copy = is_with_data_repacking() ? get_brgemm_copy() : nullptr;
    std::vector<ov::PartialShape> planar_input_shapes = {
            ngraph::snippets::utils::get_port_planar_shape(input_value(0)),
            ngraph::snippets::utils::get_port_planar_shape(brgemm_copy ? brgemm_copy->input_value(0) : input_value(1))
    };

    auto output_shape = get_output_partial_shape(planar_input_shapes);
    const auto& output_layout = ngraph::snippets::utils::get_node_output_layout(this);
    set_output_type(0,
                    get_output_type(),
                    ngraph::snippets::utils::get_reordered_planar_shape(output_shape, output_layout));

    //Additional check for 3rd input
    if (one_of(m_type, Type::WithCompensations, Type::AMX)) {
        const auto shape = get_input_partial_shape(2);
        NGRAPH_CHECK(shape.is_static(), "BRGEMM Scratch must have static shape");
        const auto type = get_input_element_type(2);
        if (is_with_compensations()) {
            const auto element_type_b = get_input_element_type(0);
            const auto shape_b = planar_input_shapes[1].get_shape();
            const auto N = *shape_b.rbegin();
            const auto N_blk = element_type_b == element::f32 ? N :
                               element_type_b == element::bf16 ? 32 : 64;
            const auto expected_shape = ov::Shape{rnd_up(N, N_blk)};
            const auto expected_type = ov::element::f32;
            NGRAPH_CHECK(expected_shape == shape.get_shape() && expected_type == type,
                         "BRGEMM Scratch with compensations must have shape {rnd_up(N, N_blk)} and FP32 element type");
        } else {
            NGRAPH_CHECK(ngraph::shape_size(shape.get_shape()) == SCRATCH_BYTE_SIZE && type == ov::element::u8,
                         "BRGEMM Scratch for space workplace must be static, have U8 element type and size is equal to " + std::to_string(SCRATCH_BYTE_SIZE));
        }
    }
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::shared_ptr<BrgemmCPU> new_node = nullptr;
    if (!is_with_scratchpad()) {
        new_node = std::make_shared<BrgemmCPU>(new_args.at(0), new_args.at(1), m_type,
                                               get_offset_a(), get_offset_b(), get_offset_c());
    } else {
        new_node = std::make_shared<BrgemmCPU>(new_args.at(0), new_args.at(1), new_args.at(2), m_type,
                                               get_offset_a(), get_offset_b(), get_offset_scratch(), get_offset_c());
    }
    return new_node;
}

std::shared_ptr<BrgemmCopyB> BrgemmCPU::get_brgemm_copy() const {
    OPENVINO_ASSERT(one_of(m_type, Type::WithDataRepacking, Type::WithCompensations, Type::AMX), "Brgemm doesn't need BrgemmCopyB");
    if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::Buffer>(get_input_node_shared_ptr(1))) {
        return ov::as_type_ptr<BrgemmCopyB>(buffer->get_input_node_shared_ptr(0));
    }
    throw ov::Exception("BrgemmCopyB hasn't been found!");
}

size_t BrgemmCPU::get_offset_scratch() const {
    OPENVINO_ASSERT(is_with_scratchpad() && get_input_size() == 3, "Offset of scratchpad must be only in Brgemm with scratchpad on 3rd input");
    return get_input_offset(2);
}

} // namespace intel_cpu
} // namespace ov