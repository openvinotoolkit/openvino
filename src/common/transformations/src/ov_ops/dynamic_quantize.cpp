// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/variadic_split.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace op {
namespace internal {

DynamicQuantize::DynamicQuantize(const Output<Node>& data, std::vector<uint64_t> group_sizes, element::Type dt_scale)
    : Op({data}),
      m_group_sizes(std::move(group_sizes)),
      m_dt_scale(dt_scale) {
    OPENVINO_ASSERT(data.get_partial_shape().rank() == m_group_sizes.size(),
                    "FC input rank should be same as the rank of group_size ",
                    data.get_tensor_ptr()->get_partial_shape().rank(),
                    " / ",
                    m_group_sizes.size());
    set_output_size(2);
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};

    auto out_shapes = shape_infer(this, input_shapes, m_group_sizes);
    set_output_type(0, element::i8, out_shapes[0]);
    set_output_type(1, m_dt_scale, out_shapes[1]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0), m_group_sizes, m_dt_scale);
}

std::vector<ov::PartialShape> DynamicQuantize::shape_infer(const DynamicQuantize* op,
                                                           const std::vector<ov::PartialShape>& input_shapes,
                                                           const std::vector<uint64_t>& group_sizes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.push_back(input_shapes[0]);

    auto scale_shape = input_shapes[0];
    OPENVINO_ASSERT(scale_shape.size() == group_sizes.size(),
                    "Scale_shape and group_size are supposed to have same rank: ",
                    scale_shape.size(),
                    " / ",
                    group_sizes.size());
    for (size_t i = 0; i < scale_shape.size(); i++) {
        if (scale_shape[i].is_dynamic())
            continue;

        if (group_sizes[i] == UINT64_MAX)
            scale_shape[i] = 1;
        else {
            scale_shape[i] /= group_sizes[i];  // if group_size is larger than shape, scale_shape will be 1
            scale_shape[i] = std::max(static_cast<int>(scale_shape[i].get_length()), 1);
        }
    }
    out_shapes.push_back(scale_shape);
    return out_shapes;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
