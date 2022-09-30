// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/embeddingbag_offsets_base.hpp"

#include "embeddingbag_offsets_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::EmbeddingBagOffsetsBase);

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index,
                                                               const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, offsets, default_index, per_sample_weights}) {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index)
    : Op({emb_table, indices, offsets, default_index}) {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets)
    : Op({emb_table, indices, offsets}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::EmbeddingBagOffsetsBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_EmbeddingBagOffsetsBase_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(OFFSETS) == element::i64 || get_input_element_type(OFFSETS) == element::i32,
        "OFFSETS type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INDICES).compatible(get_input_element_type(OFFSETS)),
                          "Offsets element type (",
                          get_input_element_type(OFFSETS),
                          ") must match indices element type (",
                          get_input_element_type(INDICES),
                          ")");

    if (get_input_size() >= 4) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(DEFAULT_INDEX) == element::i64 ||
                                  get_input_element_type(DEFAULT_INDEX) == element::i32,
                              "DEFAULT_INDEX type must be i32 or i64");

        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(INDICES).compatible(get_input_element_type(DEFAULT_INDEX)),
                              "Default_index element type (",
                              get_input_element_type(DEFAULT_INDEX),
                              ") must match indices element type (",
                              get_input_element_type(INDICES),
                              ")");
    }

    if (get_input_size() == 5) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");
    }

    element::Type result_et = get_input_element_type(EMB_TABLE);

    std::vector<PartialShape> result_shapes = {PartialShape::dynamic()};
    std::vector<PartialShape> input_shapes;
    for (int i = 0; i < get_input_size(); i++)
        input_shapes.push_back(get_input_partial_shape(i));

    shape_infer(this, input_shapes, result_shapes);

    set_output_type(0, result_et, result_shapes[0]);
}

bool ov::op::util::EmbeddingBagOffsetsBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_EmbeddingBagOffsetsBase_visit_attributes);
    return true;
}
