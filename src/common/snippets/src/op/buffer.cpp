// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/op/buffer.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Buffer::Buffer(const ov::Output<ov::Node>& arg) : Buffer(ov::OutputVector{arg}) {}

Buffer::Buffer(const OutputVector& arguments) : Op(arguments), m_impl(std::make_shared<IntermediateMemoryImpl>()) {
    constructor_validate_and_infer_types();
}
Buffer::Buffer(const ov::Shape& shape, ov::element::Type element_type) : Op(), m_impl(std::make_shared<NewMemoryImpl>(shape, element_type)) {
    constructor_validate_and_infer_types();
}
Buffer::Buffer(const OutputVector& arguments, std::shared_ptr<BaseImpl> impl) : Op(arguments), m_impl(std::move(impl)) {
    constructor_validate_and_infer_types();
}

bool Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    m_impl->visit_attributes(visitor);
    return true;
}

void Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    m_impl->validate_and_infer_types(this);
}

std::shared_ptr<Node> Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    return std::shared_ptr<Buffer>(new Buffer(new_args, m_impl->clone()));
}

Buffer::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& buffer = ov::as_type_ptr<Buffer>(n);
    OPENVINO_ASSERT(buffer, "Got invalid node in Buffer::ShapeInfer");
    m_impl_shape_infer = buffer->m_impl->get_shape_infer();
}

IShapeInferSnippets::Result Buffer::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    return m_impl_shape_infer->infer(input_shapes);
}

std::shared_ptr<Buffer::BaseImpl> Buffer::IntermediateMemoryImpl::clone() const {
    return std::make_shared<IntermediateMemoryImpl>();
}

void Buffer::IntermediateMemoryImpl::validate_and_infer_types(Buffer* buffer) const {
    OPENVINO_ASSERT(buffer, "Buffer is missed");
    OPENVINO_ASSERT(buffer->get_input_size() != 0, "IntermediateMemory Buffer must have inputs");
    const auto inputs = buffer->input_values();
    const auto& inshape = buffer->get_input_partial_shape(0);
    const auto& intype = buffer->get_input_element_type(0);
    OPENVINO_ASSERT(std::all_of(inputs.cbegin() + 1, inputs.cend(),
                                [&](const ov::Output<ov::Node>& in) { return in.get_partial_shape() == inshape && in.get_element_type() == intype; }),
                    "All inputs of Buffers must have the same shape and element type");
    buffer->set_output_type(0, intype, inshape);
}

Buffer::IntermediateMemoryImpl::ShapeInfer::Result Buffer::IntermediateMemoryImpl::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(!input_shapes.empty(), "IntermediateMemoryBuffer shape inference must have input shapes");
    return {{input_shapes[0].get()}, ShapeInferStatus::success};
}

Buffer::NewMemoryImpl::NewMemoryImpl(const ov::Shape& shape, ov::element::Type element_type)
    : m_shape(shape), m_element_type(element_type) {}

size_t Buffer::NewMemoryImpl::get_allocation_size() const {
    return ov::shape_size(m_shape);
}

std::shared_ptr<Buffer::BaseImpl> Buffer::NewMemoryImpl::clone() const {
    return std::make_shared<NewMemoryImpl>(m_shape, m_element_type);
}

void Buffer::NewMemoryImpl::validate_and_infer_types(Buffer* buffer) const {
    OPENVINO_ASSERT(buffer, "Buffer is missed");
    OPENVINO_ASSERT(buffer->get_input_size() == 0, "NewMemory Buffer mustn't have inputs");
    buffer->set_output_type(0, m_element_type, m_shape);
}

bool Buffer::NewMemoryImpl::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("shape", m_shape);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

Buffer::NewMemoryImpl::ShapeInfer::ShapeInfer(ov::Shape shape) : m_shape(std::move(shape)) {}

Buffer::NewMemoryImpl::ShapeInfer::Result Buffer::NewMemoryImpl::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.empty(), "NewMemoryBuffer shape inference must have input shapes");
    return {{m_shape}, ShapeInferStatus::success};
}

} // namespace op
} // namespace snippets
} // namespace ov
