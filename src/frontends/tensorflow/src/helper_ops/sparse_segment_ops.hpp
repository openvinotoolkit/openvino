// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class SparseSegmentSum : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("SparseSegmentSum", "ov::frontend::tensorflow::util", ov::frontend::tensorflow::InternalOperation);

    SparseSegmentSum(const Output<Node>& data,
                     const Output<Node>& indices,
                     const Output<Node>& segment_ids,
                     const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder, OutputVector{data, indices, segment_ids}, 1) {
        validate_and_infer_types();
    }

    SparseSegmentSum(const Output<Node>& data,
                     const Output<Node>& indices,
                     const Output<Node>& segment_ids,
                     const Output<Node>& num_segments,
                     const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder,
                                                      OutputVector{data, indices, segment_ids, num_segments},
                                                      1) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // SparseSegmentSum computes the sum along sparse segments of a tensor.
        // Inputs:
        // 0) data - A Tensor with data that will be assembled in the output
        // 1) indices - A 1-D Tensor with indices into data. Has same rank as segment_ids
        // 2) segment_ids - A 1-D Tensor with indices into the output Tensor. Values should be sorted and can be
        // repeated. 3) num_segments - An optional int32 scalar. Indicates the size of the output Tensor. Outputs: 0) A
        // tensor of the shape as data, except for dimension 0 which has size k, the number of segments specified via
        // num_segments or inferred for the last element in segments_ids.
        ov::PartialShape output_shape = get_input_partial_shape(0);
        auto output_rank = output_shape.rank();

        // num_segments input is optional so it is not always possible to deduce the first dimension of the output shape
        if (get_input_size() > 3) {
            ov::PartialShape num_segments_value;
            if (output_rank.is_static() && ov::evaluate_as_partial_shape(input_value(3), num_segments_value)) {
                FRONT_END_OP_CONVERSION_CHECK(output_rank.get_length() >= 1,
                                              "Data input of SparseSegmentSum must be of rank >= 1.");
                output_shape[0] = num_segments_value[0];
            }
        }

        set_output_type(0, get_input_element_type(0), output_shape);
    }
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
