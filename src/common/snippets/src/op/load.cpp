// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/load.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

Load::Load(const Output<Node>& x, const size_t count, const size_t offset) : MemoryAccess({x}, count, offset) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Load::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Load);
    check_new_args_count(this, new_args);
    return std::make_shared<Load>(new_args.at(0), m_count, m_offset);
}


LoadReshape::LoadReshape(const Output<ov::Node>& x, const size_t count, const size_t offset, std::vector<size_t> order)
                            : Load(x, count, offset), m_order(std::move(order)) {
    const auto& in_shape = x.get_partial_shape();
    NGRAPH_CHECK(in_shape.is_static(), "LoadReshape supports only static input shapes");
    const auto in_shape_size = in_shape.size();
    NGRAPH_CHECK(m_order.size() == in_shape_size, "LoadReshape got new_order of invalid size");
    NGRAPH_CHECK(*std::max_element(m_order.begin(), m_order.end()) == in_shape_size - 1 &&
                 *std::min_element(m_order.begin(), m_order.end()) == 0, "LoadReshape detected invalid values in new_order");
    const std::set<size_t> unique_dims(order.begin(), order.end());
    NGRAPH_CHECK(unique_dims.size() == order.size(), "LoadReshape order must not contain repeated elements");
    constructor_validate_and_infer_types();
}

void snippets::op::LoadReshape::validate_and_infer_types() {
    const auto& old_shape = get_input_partial_shape(0);
    ov::PartialShape new_shape;
    for (const auto idx : m_order)
        new_shape.push_back(old_shape[idx]);
    set_output_type(0, get_input_element_type(0), new_shape);
}

bool snippets::op::LoadReshape::visit_attributes(AttributeVisitor& visitor) {
    Load::visit_attributes(visitor);
    visitor.on_attribute("order", m_order);
    return true;
}

std::shared_ptr<Node> snippets::op::LoadReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LoadReshape);
    check_new_args_count(this, new_args);
    return std::make_shared<LoadReshape>(new_args.at(0), m_count, m_offset, m_order);
}

}// namespace op
}// namespace snippets
}// namespace ngraph
