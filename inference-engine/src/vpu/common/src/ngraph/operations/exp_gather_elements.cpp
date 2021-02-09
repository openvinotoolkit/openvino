// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/exp_gather_elements.hpp"

#include <ngraph/validation_util.hpp>

namespace ngraph { namespace vpu { namespace op {

NGRAPH_RTTI_DEFINITION(ExpGatherElements, "ExpGatherElements", 0);

ExpGatherElements::ExpGatherElements(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& lookupIndices,
                                     const int64_t axis,
                                     const int64_t lookupAxis)
    : ngraph::op::Op({data, indices, lookupIndices})
    , m_axis(axis)
    , m_lookup_axis(lookupAxis) {
    constructor_validate_and_infer_types();
}

void ExpGatherElements::validate_and_infer_types() {
    const auto& dataType = get_input_element_type(0);
    const auto& indicesType = get_input_element_type(1);
    const auto& lookupIndicesType = get_input_element_type(2);

    NODE_VALIDATION_CHECK(this, indicesType == element::Type_t::i32 || indicesType == element::Type_t::i64,
                          "indices must be of int32 or int64 type. But instead got: ", indicesType);
    NODE_VALIDATION_CHECK(this, lookupIndicesType == element::Type_t::i32 || lookupIndicesType == element::Type_t::i64,
                          "lookupIndices must be of int32 or int64 type. But instead got: ", lookupIndicesType);

    const auto& dataPShape = get_input_partial_shape(0);
    const auto& indicesPShape = get_input_partial_shape(1);
    const auto& lookupIndicesPShape = get_input_partial_shape(2);
    const auto& dataRank = dataPShape.rank();
    const auto& indicesRank = indicesPShape.rank();
    const auto& lookupIndicesRank = lookupIndicesPShape.rank();

    NODE_VALIDATION_CHECK(this, dataRank.is_static() && indicesRank.is_static() && lookupIndicesRank.is_static(),
                          "Dynamic rank is not supported for any input");

    const auto axis = ngraph::normalize_axis(description(), m_axis, indicesRank);
    const auto lookupAxis = ngraph::normalize_axis(description(), m_lookup_axis, dataRank);

    NODE_VALIDATION_CHECK(this, axis < indicesRank.get_length(),
                          "axis must be within interval (-indices.rank,  indices.rank - 1). But instead Got", m_axis);
    NODE_VALIDATION_CHECK(this, lookupAxis < dataRank.get_length(),
                          "lookupAxis must be within interval (-data.rank,  data.rank - 1). But instead Got", m_lookup_axis);

    set_output_type(0, dataType, indicesPShape);
}

bool ExpGatherElements::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("lookup_axis", m_lookup_axis);
    return true;
}

std::shared_ptr<Node> ExpGatherElements::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ExpGatherElements>(new_args.at(0), new_args.at(1), new_args.at(2), get_axis(), m_lookup_axis);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
