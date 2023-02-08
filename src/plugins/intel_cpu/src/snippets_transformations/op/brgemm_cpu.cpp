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

// =================================== BrgemmCPU ===================================

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B,
                     const bool transposed_a, const bool transposed_b,
                     const size_t offset_a, const size_t offset_b, const size_t offset_c)
    : Brgemm(A, B, transposed_a, transposed_b, offset_a, offset_b, offset_c) {}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCPU>(new_args.at(0), new_args.at(1),
                                       m_transposed_a, m_transposed_b,
                                       get_offset_a(), get_offset_b(), get_offset_c());
}

// ============================= BrgemmWithRepackingCPU ==============================

std::shared_ptr<BrgemmCopyBBase> BrgemmWithRepackingCPU::get_brgemm_copy() const {
    if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::IntermediateBuffer>(get_input_node_shared_ptr(1))) {
        return ov::as_type_ptr<BrgemmCopyBBase>(buffer->get_input_node_shared_ptr(0));
    }
    return nullptr;
}

void BrgemmWithRepackingCPU::validate_output() {
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "BrgemmWithRepackingCPU currently supports only static shapes.");

    const auto brgemm_copy = get_brgemm_copy();
    NODE_VALIDATION_CHECK(this, brgemm_copy, "BrgemmWithRepackingCPU requires BrgemmCopyB on second input.");

    std::vector<ov::PartialShape> planar_input_shapes = {
            ngraph::snippets::utils::get_port_planar_shape(input_value(0)),
            ngraph::snippets::utils::get_port_planar_shape(brgemm_copy->input_value(0))
    };

    auto output_shape = get_output_partial_shape(planar_input_shapes);
    const auto& output_layout = ngraph::snippets::utils::get_node_output_layout(this);
    set_output_type(0,
                    get_output_type(),
                    ngraph::snippets::utils::get_reordered_planar_shape(output_shape, output_layout));
}

// ============================= BrgemmIndependentCPU ==============================

BrgemmIndependentCPU::BrgemmIndependentCPU(const Output<Node>& A, const Output<Node>& B,
                                           const bool transposed_a, const bool transposed_b,
                                           const size_t offset_a, const size_t offset_b, const size_t offset_c)
    : BrgemmWithRepackingCPU() { // We call default ctor of Brgemm class to avoid incorrect shape infer in constructor_validate_and_type_infer() call
    set_arguments({A, B});
    set_output_size(1);
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    m_transposed_a = transposed_a;
    m_transposed_b = transposed_b;
    constructor_validate_and_infer_types();
}

void BrgemmIndependentCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmIndependentCPU_validate_and_infer_types);
    validate_output();
}


std::shared_ptr<Node> BrgemmIndependentCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmIndependentCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmIndependentCPU>(new_args.at(0), new_args.at(1),
                                                  m_transposed_a, m_transposed_b,
                                                  get_offset_a(), get_offset_b(), get_offset_c());
}

// ============================ BrgemmWithScratchCPU =============================

BrgemmWithScratchCPU::BrgemmWithScratchCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                                           const bool transposed_a, const bool transposed_b,
                                           const size_t offset_a, const size_t offset_b, const size_t offset_scratch, const size_t offset_c)
    : BrgemmWithRepackingCPU() {
    set_arguments({A, B, scratch});
    set_output_size(1);
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_input_port_descriptor({0, offset_scratch}, 2);
    set_output_port_descriptor({0, offset_c}, 0);
    m_transposed_a = transposed_a;
    m_transposed_b = transposed_b;
    constructor_validate_and_infer_types();
}

// ============================ BrgemmWithCompensationsCPU =============================

BrgemmWithCompensationsCPU::BrgemmWithCompensationsCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                                                       const bool transposed_a, const bool transposed_b,
                                                       const size_t offset_a, const size_t offset_b, const size_t offset_scratch, const size_t offset_c)
    : BrgemmWithScratchCPU(A, B, scratch, transposed_a, transposed_b, offset_a, offset_b, offset_scratch, offset_c) {
    constructor_validate_and_infer_types();
}

void BrgemmWithCompensationsCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmWithCompensationsCPU_validate_and_infer_types);
    validate_output();

    const auto brgemm_copy = get_brgemm_copy();
    const auto original_shape_b = ngraph::snippets::utils::get_port_planar_shape(brgemm_copy->input_value(0));
    const auto shape = get_input_partial_shape(2);
    NGRAPH_CHECK(shape.is_static(), "BRGEMM compensations must have static shape");
    const auto type = get_input_element_type(2);
    const auto element_type_b = get_input_element_type(0);
    const auto shape_b = original_shape_b.get_shape();
    const auto N = *shape_b.rbegin();
    const auto N_blk = element_type_b == element::f32 ? N :
                       element_type_b == element::bf16 ? 32 : 64;
    const auto expected_shape = ov::Shape{rnd_up(N, N_blk)};
    const auto expected_type = ov::element::f32;
    NGRAPH_CHECK(expected_shape == shape.get_shape() && expected_type == type,
                 "Second input of BrgemmWithCompensationsCPU must have shape {rnd_up(N, N_blk)} and FP32 element type");
}

std::shared_ptr<Node> BrgemmWithCompensationsCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmWithCompensationsCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmWithCompensationsCPU>(new_args.at(0), new_args.at(1), new_args.at(2),
                                                        m_transposed_a, m_transposed_b,
                                                        get_offset_a(), get_offset_b(), get_offset_scratch(), get_offset_c());
}


// ============================ BrgemmAMXCPU =============================

BrgemmAMXCPU::BrgemmAMXCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                           const bool transposed_a, const bool transposed_b,
                           const size_t offset_a, const size_t offset_b, const size_t offset_scratch, const size_t offset_c)
    : BrgemmWithScratchCPU(A, B, scratch, transposed_a, transposed_b, offset_a, offset_b, offset_scratch, offset_c) {
    constructor_validate_and_infer_types();
}

void BrgemmAMXCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmAMXCPU_validate_and_infer_types);
    validate_output();

    const auto shape = get_input_partial_shape(2);
    NGRAPH_CHECK(shape.is_static(), "BRGEMM Scratch must have static shape");
    const auto type = get_input_element_type(2);
    NGRAPH_CHECK(ngraph::shape_size(shape.get_shape()) == 8 * 1024 && type == ov::element::f32,
                 "BrgemmAMXCPU Scratch for space workplace must be static, have FP32 element type and 8x1024 shape size");
}

std::shared_ptr<Node> BrgemmAMXCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmAMXCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmAMXCPU>(new_args.at(0), new_args.at(1), new_args.at(2),
                                          m_transposed_a, m_transposed_b,
                                          get_offset_a(), get_offset_b(), get_offset_scratch(), get_offset_c());
}

} // namespace intel_cpu
} // namespace ov