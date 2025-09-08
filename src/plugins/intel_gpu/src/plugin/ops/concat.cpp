// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/concat.hpp"

#include "intel_gpu/primitives/concatenation.hpp"

namespace ov::intel_gpu {

static void CreateConcatOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Concat>& op) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    int64_t axis = op->get_axis();
    if (axis < 0)
        axis = axis + static_cast<int64_t>(op->get_input_partial_shape(0).rank().get_length());

    auto concatPrim = cldnn::concatenation(
        layerName,
        inputs,
        axis,
        cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    p.add_primitive(*op, concatPrim);
}

REGISTER_FACTORY_IMPL(v0, Concat);

}  // namespace ov::intel_gpu
