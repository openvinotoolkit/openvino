// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/embeddingbag_packed_base.hpp"

#include "itt.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::EmbeddingBagPackedBase);

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table,
                                                             const Output<Node>& indices,
                                                             const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, per_sample_weights}) {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table, const Output<Node>& indices)
    : Op({emb_table, indices}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::EmbeddingBagPackedBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(INDICES).is_dynamic() || get_input_partial_shape(INDICES).to_shape().size() == 2,
        "INDICES must be 2D");

    if (get_input_size() == 3) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(PER_SAMPLE_WEIGHTS).is_dynamic() ||
                                  get_input_partial_shape(PER_SAMPLE_WEIGHTS).to_shape().size() == 2,
                              "PER_SAMPLE_WEIGHTS must be 2D");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(INDICES).compatible(get_input_partial_shape(PER_SAMPLE_WEIGHTS)),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same");
    }

    element::Type result_et = get_input_element_type(EMB_TABLE);

    const PartialShape& emb_table_shape = get_input_partial_shape(EMB_TABLE);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    PartialShape result_shape;
    if (emb_table_shape.rank().is_static()) {
        result_shape = emb_table_shape;
        result_shape[0] = indices_shape.rank().is_static() ? indices_shape[0] : Dimension::dynamic();
    } else {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

bool ov::op::util::EmbeddingBagPackedBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_visit_attributes);
    return true;
}
