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

DynamicQuantize::DynamicQuantize(const Output<Node>& data, size_t group_size, element::Type dt_scale)
    : Op({data})
    , m_group_size(group_size)
    , m_dt_scale(dt_scale) {
    set_output_size(2);
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0)
    };

    auto out_shapes = shape_infer(this, input_shapes);
    set_output_type(0, element::i8, out_shapes[0]);
    set_output_type(1, m_dt_scale, out_shapes[1]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0), m_group_size, m_dt_scale);
}

std::vector<ov::PartialShape> DynamicQuantize::shape_infer(const DynamicQuantize* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.push_back(input_shapes[0]);
    // FIXME: generalize to N-dim case
    auto scale_shape = input_shapes[0];
    for (size_t i = 2; i < scale_shape.size(); i++)
        scale_shape[i] = 1;
    out_shapes.push_back(scale_shape);
    return out_shapes;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
