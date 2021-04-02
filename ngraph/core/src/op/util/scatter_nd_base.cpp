// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/scatter_nd_base.hpp"
#include "itt.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::util::ScatterNDBase::type_info;
constexpr int op::util::ScatterNDBase::INPUTS;
constexpr int op::util::ScatterNDBase::INDICES;
constexpr int op::util::ScatterNDBase::UPDATES;

op::util::ScatterNDBase::ScatterNDBase(const Output<Node>& data,
                                       const Output<Node>& indices,
                                       const Output<Node>& updates)
    : Op({data, indices, updates})
{
    constructor_validate_and_infer_types();
}

bool op::util::ScatterNDBase::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(util_ScatterNDBase_visit_attributes);
    return true;
}

void op::util::ScatterNDBase::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_ScatterNDBase_validate_and_infer_types);
    element::Type inputs_et = get_input_element_type(INPUTS);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    const PartialShape& inputs_shape = get_input_partial_shape(INPUTS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);
    const PartialShape& updates_shape = get_input_partial_shape(UPDATES);

    const auto& inputs_rank = inputs_shape.rank();
    const auto& indices_rank = indices_shape.rank();
    const auto& updates_rank = updates_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(
        this, updates_et == inputs_et, "Updates element type must be the same as inputs");

    NODE_VALIDATION_CHECK(this,
                          indices_rank.is_dynamic() || indices_rank.get_length() >= 1,
                          "Indices rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          inputs_rank.is_dynamic() || indices_rank.is_dynamic() ||
                              indices_shape[indices_rank.get_length() - 1].get_length() <=
                                  inputs_rank.get_length(),
                          "Last dimension of indices can be at most the rank of inputs");

    if (inputs_rank.is_static() && indices_rank.is_static() && updates_rank.is_static())
    {
        auto expected_updates_rank = indices_rank.get_length() + inputs_rank.get_length() -
                                     indices_shape[indices_rank.get_length() - 1].get_length() - 1;
        // If expected updates rank is 0D it also can be a tensor with one element
        NODE_VALIDATION_CHECK(
            this,
            updates_rank.get_length() == expected_updates_rank || expected_updates_rank == 0,
            "Rank of updates must be rank of inputs + rank of indices - last dimension of indices "
            "- 1");

        bool compatible = true;
        if (inputs_shape.is_static() && indices_shape.is_static() && updates_shape.is_static())
        {
            size_t static_indices_rank = indices_rank.get_length();
            for (size_t i = 0; i < static_indices_rank - 1; i++)
            {
                compatible = compatible && updates_shape[i].same_scheme(indices_shape[i]);
                NODE_VALIDATION_CHECK(
                    this,
                    compatible,
                    "updates_shape[0:indices_rank-1] shape must be indices_shape[:-1]");
            }
            size_t j = indices_shape[static_indices_rank - 1].get_length();
            for (size_t i = static_indices_rank - 1; i < expected_updates_rank; i++, j++)
            {
                compatible = compatible && updates_shape[i].same_scheme(inputs_shape[j]);
                NODE_VALIDATION_CHECK(
                    this,
                    compatible,
                    "updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]");
            }
        }
    }

    set_output_type(0, inputs_et, inputs_shape);
}
