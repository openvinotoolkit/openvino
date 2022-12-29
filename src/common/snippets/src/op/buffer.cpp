// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/buffer.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"


using namespace std;
using namespace ngraph;

auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + static_cast<int32_t>(shape_rank) : allocation_rank;
}

size_t ngraph::snippets::op::Buffer::get_byte_size() const {
    const auto pshape = get_allocation_shape();
    // TODO: Add support of dynamism
    NGRAPH_CHECK(pshape.is_static(), "Buffer should have static shapes for memory allocation");
    const auto shape = pshape.get_shape();
    return ngraph::shape_size(shape) * get_element_type().size();
}

snippets::op::AllocationBuffer::AllocationBuffer(const Output<Node>& shape, const ov::element::Type element_type)
    : Buffer(), m_element_type(element_type) {
    set_arguments({shape});
    constructor_validate_and_infer_types();
}

bool snippets::op::AllocationBuffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(AllocationBuffer_visit_attributes);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

std::shared_ptr<Node> snippets::op::AllocationBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(AllocationBuffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AllocationBuffer>(new_args.at(0), m_element_type);
}

void snippets::op::AllocationBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AllocationBuffer_validate_and_infer_types);
    set_output_type(0, m_element_type, get_allocation_shape());
}

ov::PartialShape ngraph::snippets::op::AllocationBuffer::get_allocation_shape() const {
    ov::PartialShape shape = ov::PartialShape::dynamic();
    const auto shape_constant = ov::as_type_ptr<ngraph::op::v0::Constant>(get_input_node_shared_ptr(0));
    if (shape_constant) {
        NGRAPH_CHECK(shape_constant->get_element_type() == ov::element::i32,
                     "The AllocationBuffer expects Constant with shape of I32 element type");
        const auto dims = shape_constant->cast_vector<int32_t>();
        NGRAPH_CHECK(!dims.empty(), "The AllocationBuffer got invalid shape Constant");
        shape = ov::PartialShape(ov::Shape(std::vector<size_t>(dims.begin(), dims.end())));
    }
    return shape;
}

snippets::op::IntermediateBuffer::IntermediateBuffer(const ov::Output<ov::Node>& x) : Buffer() {
    set_arguments({x});
    constructor_validate_and_infer_types();
}

snippets::op::IntermediateBuffer::IntermediateBuffer(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& shape) : Buffer() {
    set_arguments({x, shape});
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> snippets::op::IntermediateBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(IntermediateBuffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<IntermediateBuffer>(new_args.at(0), new_args.at(1));
    } else if (new_args.size() == 1) {
        return std::make_shared<IntermediateBuffer>(new_args.at(0));
    }

    throw ngraph_error("The IntermediateBuffer op got invalid input count");
}

void snippets::op::IntermediateBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(IntermediateBuffer_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

ov::PartialShape ngraph::snippets::op::IntermediateBuffer::get_allocation_shape() const {
    if (get_input_size() == 1) {
        return get_input_partial_shape(0);
    }

    const auto shape_constant = ov::as_type_ptr<ngraph::op::v0::Constant>(get_input_node_shared_ptr(1));
    if (shape_constant) {
        NGRAPH_CHECK(shape_constant->get_element_type() == ov::element::i32,
                     "The AllocationBuffer expects Constant with shape of I32 element type");
        const auto dims = shape_constant->cast_vector<int32_t>();
        NGRAPH_CHECK(!dims.empty(), "The AllocationBuffer got invalid shape Constant");
        return ov::PartialShape(ov::Shape(std::vector<size_t>(dims.begin(), dims.end())));
    }
    return ov::PartialShape::dynamic();
}

std::shared_ptr<ov::Node> ngraph::snippets::op::IntermediateBuffer::create_shape_constant(const ov::PartialShape& shape, size_t allocation_rank) {
    if (shape.rank().is_dynamic())
        return nullptr;
    const auto normalize_rank = utils::normalize_rank(static_cast<int32_t>(allocation_rank), shape.size());
    const auto offset = static_cast<int32_t>(shape.size()) - normalize_rank;
    return create_shape_constant(ov::PartialShape(std::vector<ov::Dimension>{shape.begin() + offset, shape.end()}));
}

std::shared_ptr<ov::Node> ngraph::snippets::op::IntermediateBuffer::create_shape_constant(const ov::PartialShape& shape) {
    if (shape.is_dynamic())
        return nullptr;
    return std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{shape.size()}, shape.get_shape());
}