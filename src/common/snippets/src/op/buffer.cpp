// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/op/buffer.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Buffer::Buffer(const OutputVector& arguments, size_t allocation_size, size_t reg_group, size_t cluster_id)
    : Op(arguments), m_allocation_size(allocation_size), m_reg_group(reg_group), m_cluster_id(cluster_id), m_offset(0) {
    constructor_validate_and_infer_types();
}

bool Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    auto element_type = get_element_type();
    auto allocation_size = utils::value2str(m_allocation_size);
    auto offset = utils::value2str(m_offset);
    visitor.on_attribute("allocation_size", allocation_size);
    visitor.on_attribute("offset", offset);
    visitor.on_attribute("reg_group", m_reg_group);
    visitor.on_attribute("cluster_id", m_cluster_id);
    visitor.on_attribute("element_type", element_type);
    return true;
}

bool Buffer::is_defined() const {
    return !utils::is_dynamic_value(m_allocation_size);
}

size_t Buffer::get_byte_size() const {
    if (is_defined())
        return m_allocation_size * get_element_type().size();
    return utils::get_dynamic_value<size_t>();
}

IntermediateMemoryBuffer::IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, size_t allocation_size, size_t reg_group, size_t cluster_id)
    : Buffer({arg}, allocation_size, reg_group, cluster_id) {
    constructor_validate_and_infer_types();
}

void IntermediateMemoryBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    ov::PartialShape output_shape;
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> IntermediateMemoryBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<IntermediateMemoryBuffer>(new_args.at(0), m_allocation_size, m_reg_group, m_cluster_id);
    new_buffer->set_offset(m_offset);
    return new_buffer;
}

NewMemoryBuffer::NewMemoryBuffer(const ov::Shape& shape, size_t reg_group, size_t cluster_id, ov::element::Type element_type)
    : Buffer({}, ov::shape_size(shape), reg_group, cluster_id), m_output_shape(shape), m_element_type(element_type) {
    constructor_validate_and_infer_types();
}

void NewMemoryBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 0, "Buffer with new allocated memory mustn't have arguments!");
    set_output_type(0, m_element_type, m_output_shape);
}

std::shared_ptr<Node> NewMemoryBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<NewMemoryBuffer>(m_output_shape, m_reg_group, m_cluster_id, m_element_type);
    new_buffer->set_offset(m_offset);
    return new_buffer;
}

void NewMemoryBuffer::set_element_type(ov::element::Type element_type) {
    m_element_type = std::move(element_type);
    // Apply the change
    validate_and_infer_types();
}

NewMemoryBuffer::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& buffer = ov::as_type_ptr<NewMemoryBuffer>(n);
    OPENVINO_ASSERT(buffer, "Got invalid node in NewMemoryBuffer::ShapeInfer");
    m_shape = buffer->get_shape();
}

IShapeInferSnippets::Result NewMemoryBuffer::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.empty(), "NewMemoryBuffer shape inference mustn't have input shapes");
    return {{m_shape}, ShapeInferStatus::success};
}

} // namespace op
} // namespace snippets
} // namespace ov
