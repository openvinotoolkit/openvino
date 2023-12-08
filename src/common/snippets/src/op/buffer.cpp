// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/op/buffer.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Buffer::Buffer(const OutputVector& arguments, const ov::Shape& shape, size_t id, ov::element::Type element_type)
    : Op(arguments), m_shape(shape), m_id(id), m_element_type(std::move(element_type)), m_offset(0) {
    constructor_validate_and_infer_types();
}

bool Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    visitor.on_attribute("allocation_shape", m_shape);
    visitor.on_attribute("offset", m_offset);
    visitor.on_attribute("id", m_id);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

size_t Buffer::get_byte_size() const {
    const auto shape = get_allocation_shape();
    return ov::shape_size(shape) * m_element_type.size();
}

IntermediateMemoryBuffer::IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape, size_t id)
    : Buffer({arg}, shape, id) {
    constructor_validate_and_infer_types();
}

IntermediateMemoryBuffer::IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank, size_t id)
    : Buffer({arg}, compute_shape_from_allocation_rank(arg, allocation_rank), id) {
    constructor_validate_and_infer_types();
}

ov::Shape IntermediateMemoryBuffer::compute_shape_from_allocation_rank(const ov::Output<ov::Node>& arg, int32_t allocation_rank) {
    const auto& pshape = arg.get_partial_shape();
    OPENVINO_ASSERT(pshape.is_static(), "Buffer supports only static input shape");
    const auto shape = pshape.get_shape();
    const auto normalize_rank = utils::normalize_rank(static_cast<int32_t>(allocation_rank), shape.size());
    const auto offset = static_cast<int32_t>(shape.size()) - normalize_rank;
    return ov::Shape{shape.begin() + offset, shape.end()};
}

void IntermediateMemoryBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    ov::PartialShape output_shape;
    m_element_type = get_input_element_type(0);
    output_shape = get_input_partial_shape(0);
    set_output_type(0, m_element_type, output_shape);
}

std::shared_ptr<Node> IntermediateMemoryBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<IntermediateMemoryBuffer>(new_args.at(0), m_shape, m_id);
    new_buffer->set_offset(m_offset);
    return new_buffer;
}

NewMemoryBuffer::NewMemoryBuffer(const ov::Shape& shape, size_t id, ov::element::Type element_type)
    : Buffer({}, shape, id, element_type) {
    constructor_validate_and_infer_types();
}

void NewMemoryBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 0, "Buffer with new allocated memory mustn't have arguments!");
    set_output_type(0, m_element_type, m_shape);
}

std::shared_ptr<Node> NewMemoryBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<NewMemoryBuffer>(m_shape, m_id, m_element_type);
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
