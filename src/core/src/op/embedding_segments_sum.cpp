// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embedding_segments_sum.hpp"

#include "embedding_segments_sum_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments,
                                                   const Output<Node>& default_index,
                                                   const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights}) {
    constructor_validate_and_infer_types();
}

op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments,
                                                   const Output<Node>& default_index)
    : Op({emb_table, indices, segment_ids, num_segments, default_index}) {
    constructor_validate_and_infer_types();
}

op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments)
    : Op({emb_table, indices, segment_ids, num_segments}) {
    constructor_validate_and_infer_types();
}

void op::v3::EmbeddingSegmentsSum::validate_and_infer_types() {
    OV_OP_SCOPE(v3_EmbeddingSegmentsSum_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(SEGMENT_IDS) == element::i64 || get_input_element_type(SEGMENT_IDS) == element::i32,
        "SEGMENT_IDS type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(NUM_SEGMENTS) == element::i64 || get_input_element_type(NUM_SEGMENTS) == element::i32,
        "NUM_SEGMENTS type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INDICES).compatible(get_input_element_type(SEGMENT_IDS)),
                          "Segment_ids element type (",
                          get_input_element_type(SEGMENT_IDS),
                          ") must match indices element type (",
                          get_input_element_type(INDICES),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(SEGMENT_IDS).compatible(get_input_element_type(NUM_SEGMENTS)),
                          "Num_segments element type (",
                          get_input_element_type(NUM_SEGMENTS),
                          ") must match Segment_ids element type (",
                          get_input_element_type(SEGMENT_IDS),
                          ")");

    if (get_input_size() >= 5) {
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

    if (get_input_size() == 6) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");
    }
    const auto& result_et = get_input_element_type(EMB_TABLE);
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto result_shapes = shape_infer(this, input_shapes);

    if (result_shapes[EMB_TABLE].rank().is_dynamic() || result_shapes[EMB_TABLE][0].is_dynamic()) {
        set_input_is_relevant_to_shape(NUM_SEGMENTS, true);
    }
    set_output_type(0, result_et, result_shapes[0]);
}

std::shared_ptr<Node> op::v3::EmbeddingSegmentsSum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_EmbeddingSegmentsSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<op::v3::EmbeddingSegmentsSum>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3));
    } else if (new_args.size() == 5) {
        return std::make_shared<op::v3::EmbeddingSegmentsSum>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3),
                                                              new_args.at(4));
    } else if (new_args.size() == 6) {
        return std::make_shared<op::v3::EmbeddingSegmentsSum>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3),
                                                              new_args.at(4),
                                                              new_args.at(5));
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
}  // namespace ov
