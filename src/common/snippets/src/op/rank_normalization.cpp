// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/rank_normalization.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

RankNormalization::RankNormalization(const Output<Node>& data, size_t num_prepend, size_t num_append) :
    ShapeInferOp({data}), m_num_prepend(num_prepend), m_num_append(num_append) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> RankNormalization::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<RankNormalization>(new_args[0], m_num_prepend, m_num_append);
}

void RankNormalization::validate_and_infer_types() {
    auto new_shape = get_input_partial_shape(0);
    // Note: other values are not allowed, only planar + blocked layout combination can be normalized.
    NODE_VALIDATION_CHECK(this, utils::one_of(m_num_append, 0lu, 1lu),
                          "num_append could be only 0 or 1, other values are not allowed.");
    new_shape.insert(new_shape.begin(), m_num_prepend, Dimension(1));
    new_shape.insert(new_shape.end(), m_num_append, Dimension(1));
    set_output_type(0, get_input_element_type(0), new_shape);
}

bool RankNormalization::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("num_prepend", m_num_prepend);
    visitor.on_attribute("num_append", m_num_append);
    return true;
}

RankNormalization::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& rank_norm = as_type_ptr<RankNormalization>(n);
    OPENVINO_ASSERT(rank_norm, "Invalid operation passed to RankNormalization::ShapeInfer: ", n->get_type_info().name);
    m_num_append = rank_norm->m_num_append;
    m_num_prepend = rank_norm->m_num_prepend;
}

IShapeInferSnippets::Result
RankNormalization::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of input shapes passed to RankNormalization::ShapeInfer::infer");
    VectorDims out_shape = input_shapes[0].get();
    out_shape.insert(out_shape.begin(), m_num_prepend, 1);
    out_shape.insert(out_shape.end(), m_num_append, 1);
    return {{out_shape}, ShapeInferStatus::success};
}

} // namespace op
} // namespace snippets
} // namespace ov