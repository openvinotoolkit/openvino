// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/op/buffer.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace op {


Buffer::Buffer(const ov::Shape& shape, ov::element::Type element_type, size_t id)
    : Op(), m_type(Type::NewMemory), m_shape(shape), m_offset(0), m_id(id), m_element_type(std::move(element_type)) {
    constructor_validate_and_infer_types();
}

Buffer::Buffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape, size_t id)
    : Op({arg}), m_type(Type::IntermediateMemory), m_shape(shape), m_offset(0), m_id(id) {
    constructor_validate_and_infer_types();
}

Buffer::Buffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank, size_t id)
    : Op({arg}), m_type(Type::IntermediateMemory), m_offset(0), m_id(id) {
    const auto& pshape = arg.get_partial_shape();
    OPENVINO_ASSERT(pshape.is_static(), "Buffer supports only static input shape");
    const auto shape = pshape.get_shape();
    const auto normalize_rank = utils::normalize_rank(static_cast<int32_t>(allocation_rank), shape.size());
    const auto offset = static_cast<int32_t>(shape.size()) - normalize_rank;
    m_shape = {shape.begin() + offset, shape.end()};
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

void Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    ov::PartialShape output_shape;
    if (m_type == Type::NewMemory) {
        OPENVINO_ASSERT(get_input_size() == 0, "Buffer with new allocated memory must to not have arguments!");
        output_shape = m_shape;
    } else if (m_type == Type::IntermediateMemory) {
        m_element_type = get_input_element_type(0);
        output_shape = get_input_partial_shape(0);
    } else {
        OPENVINO_THROW("Buffer supports only the following types: NewMemory and IntermediateMemory");
    }
    set_output_type(0, m_element_type, output_shape);
}

std::shared_ptr<Node> Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::shared_ptr<op::Buffer> new_buffer = nullptr;
    if (m_type == Type::NewMemory) {
        new_buffer = std::make_shared<Buffer>(m_shape, m_element_type, m_id);
    } else if (m_type == Type::IntermediateMemory) {
        new_buffer = std::make_shared<Buffer>(new_args.at(0), m_shape, m_id);
    } else {
        OPENVINO_THROW("Buffer supports only the following types: NewMemory and IntermediateMemory");
    }
    new_buffer->m_offset = m_offset;
    return new_buffer;
}

size_t Buffer::get_byte_size() const {
    const auto shape = get_allocation_shape();
    return ov::shape_size(shape) * get_element_type().size();
}

void Buffer::set_element_type(ov::element::Type element_type) {
    OPENVINO_ASSERT(is_new_memory(), "Only Buffer with NewMemory can change his output precision!");
    m_element_type = std::move(element_type);
    // Apply the change
    validate_and_infer_types();
}

} // namespace op
} // namespace snippets
} // namespace ov
