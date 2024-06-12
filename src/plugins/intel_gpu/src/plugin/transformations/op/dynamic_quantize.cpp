// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/variadic_split.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

DynamicQuantize::DynamicQuantize(const Output<Node>& data)
    : Op({data}) {
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    auto output_type = ov::element::Type_t::i8;

    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0)
    };

    set_output_type(0, output_type, shape_infer(this, input_shapes)[0]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0));
}

std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op, std::vector<ov::PartialShape> input_shapes) {
    return {input_shapes[0]};
}


}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
