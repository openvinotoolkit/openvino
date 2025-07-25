// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fa.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

FACPU::FACPU() : m_config(FAConfig{}) {}

FACPU::FACPU(const ov::OutputVector& inputs,
             const FAConfig& config,
             const std::vector<PortDescriptor>& input_descs,
             const PortDescriptor& output_desc,
             const std::vector<size_t>& layout_a,
             const std::vector<size_t>& layout_b,
             const std::vector<size_t>& layout_c,
             const std::vector<size_t>& layout_d) : m_config(config) {
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
  
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c, layout_d);
}

void FACPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(FACPU_validate_and_infer_types);

    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1), input(2)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, ov::element::f32, get_planar_output_shape(output_shape));
}

void FACPU::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                        const std::vector<size_t>& layout_b,
                                                        const std::vector<size_t>& layout_c,
                                                        const std::vector<size_t>& layout_d) {
    INTERNAL_OP_SCOPE(FACPU_constructor_validate_and_infer_types);

    const std::vector<ov::PartialShape> planar_input_shapes{
        snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
        snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b),
        snippets::utils::get_planar_pshape(get_input_partial_shape(2), layout_c)};
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, ov::element::f32, snippets::utils::get_planar_pshape(output_shape, layout_c));
}

std::shared_ptr<Node> FACPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FACPU_clone_with_new_inputs);

    std::vector<PortDescriptor> input_port_descriptors;
    for (size_t i = 0; i < get_input_size(); ++i) {
        input_port_descriptors.push_back(get_input_port_descriptor(i));
    }

    return std::make_shared<FACPU>(
        new_args,
        m_config,
        input_port_descriptors,
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(2))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

}  // namespace ov::intel_cpu
