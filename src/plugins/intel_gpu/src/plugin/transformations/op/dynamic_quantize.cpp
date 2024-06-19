// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/variadic_split.hpp"
#include "variadic_split_shape_inference.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

DynamicQuantize::DynamicQuantize(const Output<Node>& data)
    : Op({data}) {
    set_output_size(2);
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0)
    };

    auto out_shapes = shape_infer(this, input_shapes);
    set_output_type(0, ov::element::Type_t::i8, out_shapes[0]);
    set_output_type(1, ov::element::Type_t::f16, out_shapes[1]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0));
}

std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op, std::vector<ov::PartialShape> input_shapes) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.push_back(input_shapes[0]);
    // FIXME: generalize to N-dim case
    auto scale_shape = input_shapes[0];
    GPU_DEBUG_IF(debug_config->enable_kv_cache_compression != 1) { // per-token compression
        scale_shape[2] = 1;
    }
    scale_shape[3] = 1;
    out_shapes.push_back(scale_shape);
    return out_shapes;
}


}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
