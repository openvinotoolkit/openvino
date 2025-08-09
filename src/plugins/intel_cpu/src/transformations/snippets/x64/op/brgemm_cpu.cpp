// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <cassert>
#include <common/c_types_map.hpp>
#include <common/primitive_attr.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "common/primitive_hashing_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu {
using namespace brgemm_utils;

BrgemmCPU::BrgemmCPU() : m_config(BrgemmConfig{}) {}

BrgemmCPU::BrgemmCPU(const ov::OutputVector& inputs,
                     BrgemmConfig config,
                     const std::vector<PortDescriptor>& input_descs,
                     const PortDescriptor& output_desc,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c,
                     PostopsConfig post_ops)
    : m_config(config),
      m_post_ops_config(std::move(post_ops)),
      m_gemm_inputs_count(m_config.with_scratchpad() ? 3 : 2) {
    set_arguments(inputs);
    set_output_size(1);

    std::set<size_t> input_memory_access_ports;
    for (size_t i = 0; i < inputs.size(); ++i) {
        input_memory_access_ports.insert(i);
    }
    ctor_initialize(input_memory_access_ports, std::set<size_t>{0});

    if (!input_descs.empty()) {
        OPENVINO_ASSERT(input_descs.size() == inputs.size(),
                        "Count of input descriptors must be equal to count of inputs");
        for (size_t i = 0; i < input_descs.size(); ++i) {
            set_input_port_descriptor(input_descs[i], i);
        }
    } else {
        for (size_t i = 0; i < inputs.size(); ++i) {
            set_input_port_descriptor({0, 0}, i);
        }
    }
    set_output_port_descriptor(output_desc, 0);
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

void BrgemmCPU::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                            const std::vector<size_t>& layout_b,
                                                            const std::vector<size_t>& layout_c) {
    INTERNAL_OP_SCOPE(BrgemmCPU_constructor_validate_and_infer_types);
    validate_inputs_size();

    const std::vector<ov::PartialShape> planar_input_shapes{
        snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
        snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b)};
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), snippets::utils::get_planar_pshape(output_shape, layout_c));

    validate_with_scratchpad();
    validate_postop_inputs();
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    validate_inputs_size();

    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));

    validate_with_scratchpad();
    validate_postop_inputs();
}

void BrgemmCPU::validate_with_scratchpad() const {
    // Additional check for 3rd input
    if (m_config.with_compensations()) {
        OPENVINO_ASSERT(get_input_element_type(2) == ov::element::f32,
                        "BRGEMM Scratch with compensations must have FP32 element type");
    } else if (m_config.is_amx()) {
        OPENVINO_ASSERT(get_input_partial_shape(2).is_static(), "BRGEMM Scratch must have static shape");
        OPENVINO_ASSERT(get_input_element_type(2) == ov::element::u8, "BRGEMM Scratch must have U8 element type");
    }
}

void BrgemmCPU::validate_inputs_size() const {
    OPENVINO_ASSERT(get_input_size() >= m_gemm_inputs_count,
                    "BrgemmCPU expects at least ",
                    m_gemm_inputs_count,
                    " inputs whereas it got ",
                    get_input_size(),
                    " inputs");
}

void BrgemmCPU::validate_postop_inputs() const {
    auto result_shape = get_output_partial_shape(0);
    for (size_t i = m_gemm_inputs_count; i < get_input_size(); ++i) {
        OPENVINO_ASSERT(get_input_element_type(i) == ov::element::f32,
                        "BrgemmCPU supports only must have f32 element type but got ",
                        get_input_element_type(i),
                        " on input ",
                        i);
        const auto& input_shape = get_input_partial_shape(i);
        OPENVINO_ASSERT(
            ov::PartialShape::broadcast_merge_into(result_shape, input_shape, ov::op::AutoBroadcastType::NUMPY),
            "BrgemmCPU postop input ",
            i,
            " is not broadcastable to the output shape.");
    }
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    // Note: all brgemm inputs are Memory Access, so we can form planar vector of the MA descs from the PortMap
    std::vector<PortDescriptor> input_port_descriptors;
    for (size_t i = 0; i < get_input_size(); ++i) {
        input_port_descriptors.push_back(get_input_port_descriptor(i));
    }

    return std::make_shared<BrgemmCPU>(
        new_args,
        m_config,
        input_port_descriptors,
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout(),
        m_post_ops_config);
}

size_t BrgemmCPU::get_offset_scratch() const {
    OPENVINO_ASSERT(m_config.with_scratchpad() && m_gemm_inputs_count == 3,
                    "Offset of scratchpad must be only in Brgemm with scratchpad on 3rd input");
    return get_input_offset(2);
}

bool BrgemmCPU::visit_attributes(AttributeVisitor& visitor) {
    MemoryAccess::visit_attributes(visitor);
    auto config = m_config;
    visitor.on_attribute("BrgemmConfig", config);
    m_post_ops_config.visit_attributes(visitor);
    return true;
}

ov::element::Type BrgemmCPU::get_output_type() const {
    return m_post_ops_config.forced_output_type.value_or(Brgemm::get_output_type());
}

ov::OutputVector BrgemmCPU::get_postop_inputs() const {
    const auto& input_values = this->input_values();
    return {input_values.begin() + m_gemm_inputs_count, input_values.end()};
}

void BrgemmCPU::force_output_type(const ov::element::Type& type) {
    m_post_ops_config.forced_output_type = type;
    // Since postops config may force output type, need to reset it
    set_output_type(0, get_output_type(), get_output_partial_shape(0));
}

void BrgemmCPU::add_scalar_eltwise_postop(dnnl::impl::alg_kind_t alg_kind, float alpha, float beta) {
    OPENVINO_ASSERT(m_post_ops_config.post_ops.append_eltwise(1.F, alg_kind, alpha, beta) == dnnl_success,
                    "Failed to append scalar eltwise to brgemm postops. Alpha = ",
                    alpha,
                    " Beta = ",
                    beta);
}

void BrgemmCPU::add_binary_eltwise_postop(dnnl::impl::alg_kind_t alg_kind,
                                          const dnnl::memory::desc& desc,
                                          const ov::Output<Node>& postop_input,
                                          const size_t binary_postop_offset) {
    OPENVINO_ASSERT(m_post_ops_config.post_ops.append_binary(alg_kind, desc.get()) == dnnl_success,
                    "Failed to append binary eltwise input to brgemm postops: ",
                    postop_input);
    if (!m_post_ops_config.binary_postops_offset) {
        m_post_ops_config.binary_postops_offset = binary_postop_offset;
    }
    add_postop_input(postop_input);
}

void BrgemmCPU::add_postop_input(const ov::Output<Node>& postop_input) {
    const size_t input_idx = get_input_size();
    set_argument(input_idx, postop_input);
    m_input_ports[input_idx] = {0, 0};
    m_input_ports[input_idx].index = input_idx;

    auto& rt_info = get_rt_info();
    const auto& found = rt_info.find(ov::snippets::lowered::PortDescriptorVectorAttribute::get_type_info_static());
    // if PortDesc vectors are already created, a new postop input must be added to the input vector
    if (found != rt_info.end()) {
        auto& in_descs = found->second.as<ov::snippets::lowered::PortDescriptorVectorAttribute>().inputs;
        in_descs.emplace_back(std::make_shared<ov::snippets::lowered::PortDescriptor>(input(input_idx)));
    }
    validate_postop_inputs();
}

BrgemmCPU::PostopsConfig::PostopsConfig() : binary_postops_offset(std::nullopt), forced_output_type(std::nullopt) {}

bool BrgemmCPU::PostopsConfig::visit_attributes(AttributeVisitor& visitor) {
    auto postops_hash = dnnl::impl::primitive_hashing::get_post_op_hash(0, post_ops);
    visitor.on_attribute("postops_hash", postops_hash);
    if (binary_postops_offset) {
        visitor.on_attribute("binary_postops_offset", binary_postops_offset.value());
    }
    if (forced_output_type) {
        visitor.on_attribute("forced_output_type", forced_output_type.value());
    }
    return true;
}
}  // namespace ov::intel_cpu
