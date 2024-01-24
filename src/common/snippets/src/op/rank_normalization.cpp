// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/rank_normalization.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

RankNormalization::RankNormalization(const Output<Node>& data, size_t num_prepend, size_t num_append, size_t num_pop) :
    Op({data}), m_num_prepend(num_prepend), m_num_append(num_append), m_num_pop(num_pop) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> RankNormalization::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<RankNormalization>(new_args[0], m_num_prepend, m_num_append, m_num_pop);
}

void RankNormalization::validate_and_infer_types() {
    const auto& in_shape = get_input_partial_shape(0);
    // Note: other values are not allowed, only planar + blocked layout combination can be normalized.
    NODE_VALIDATION_CHECK(this, utils::one_of(m_num_append, 0lu, 1lu),
                          "num_append could be only 0 or 1, other values are not allowed.");
    NODE_VALIDATION_CHECK(this, utils::implication(m_num_pop > 0, in_shape.rank().is_static() && m_num_pop <= in_shape.size()),
                          "num_pop must be less rank of the original shape.");
    NODE_VALIDATION_CHECK(this, !((m_num_append > 0 || m_num_prepend > 0) && m_num_pop > 0),
                          "cannot prepend/append and pop dimensions at the same time.");
    NODE_VALIDATION_CHECK(this, std::all_of(in_shape.cbegin(), in_shape.cbegin() + m_num_pop, [](const Dimension& d) { return d == 1; }),
                          "dimension for removing must equal to 1.");
    // ov::PartialShape doesn't have API to remove dims
    ov::PartialShape new_shape;
    for (size_t i = m_num_pop; i < in_shape.size(); i++)
        new_shape.push_back(in_shape[i]);
    new_shape.insert(new_shape.begin(), m_num_prepend, Dimension(1));
    new_shape.insert(new_shape.end(), m_num_append, Dimension(1));
    set_output_type(0, get_input_element_type(0), new_shape);
}

bool RankNormalization::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("num_prepend", m_num_prepend);
    visitor.on_attribute("num_append", m_num_append);
    visitor.on_attribute("num_pop", m_num_pop);
    return true;
}

RankNormalization::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& rank_norm = as_type_ptr<RankNormalization>(n);
    OPENVINO_ASSERT(rank_norm, "Invalid operation passed to RankNormalization::ShapeInfer: ", n->get_type_info().name);
    m_num_append = rank_norm->m_num_append;
    m_num_prepend = rank_norm->m_num_prepend;
    m_num_pop = rank_norm->m_num_pop;
    m_is_shrink = m_num_pop > 0;
}

IShapeInferSnippets::Result
RankNormalization::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of input shapes passed to RankNormalization::ShapeInfer::infer");
    VectorDims out_shape = input_shapes[0].get();
    if (m_is_shrink) {
        out_shape.erase(out_shape.cbegin(), out_shape.cbegin() + m_num_pop);
    } else {
        out_shape.insert(out_shape.begin(), m_num_prepend, 1);
        out_shape.insert(out_shape.end(), m_num_append, 1);
    }
    return {{out_shape}, ShapeInferStatus::success};
}

} // namespace op
} // namespace snippets
} // namespace ov