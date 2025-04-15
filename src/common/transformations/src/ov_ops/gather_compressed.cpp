// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/gather_compressed.hpp"

#include "gather_shape_inference.hpp"

namespace ov {
namespace op {
namespace internal {

GatherCompressed::GatherCompressed(const ov::Output<Node>& data,
                                   const ov::Output<Node>& indices,
                                   const ov::Output<Node>& axis,
                                   const int64_t batch_dims,
                                   const ov::Output<Node>& decompression_scale,
                                   const ov::Output<Node>& decompression_zero_point)
    : ov::op::v8::Gather({data, indices, axis, batch_dims}) {
    set_argument(3, decompression_scale);
    set_argument(4, decompression_zero_point);
    validate_and_infer_types();
}

GatherCompressed::GatherCompressed(const ov::Output<Node>& data,
                                   const ov::Output<Node>& indices,
                                   const ov::Output<Node>& axis,
                                   const int64_t batch_dims,
                                   const ov::Output<Node>& decompression_scale)
    : ov::op::v8::Gather({data, indices, axis, batch_dims}) {
    set_argument(3, decompression_scale);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> GatherCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 4)
        return std::make_shared<GatherCompressed>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  m_batch_dims,
                                                  new_args.at(3));
    else if (new_args.size() == 5)
        return std::make_shared<GatherCompressed>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  m_batch_dims,
                                                  new_args.at(3),
                                                  new_args.at(4));
    else
        OPENVINO_THROW("Unexpected inputs count for GatherCompressed op: ", new_args.size());
}

void GatherCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size >= 4,
                          "Number of inputs is incorrect. Current value is: ",
                          input_size,
                          ", expected at least 4.");

    auto out_shapes = ov::op::shape_infer(this,
                                          std::vector<ov::PartialShape>{get_input_partial_shape(0),
                                                                        get_input_partial_shape(1),
                                                                        get_input_partial_shape(2)});
    // GatherCompressed = gather + decompression, the output precision is the same as the scale's one.
    auto output_type = get_input_element_type(3);
    set_output_type(0, output_type, out_shapes[0]);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
