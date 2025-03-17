// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/shape_of.hpp"

#include "intel_gpu/primitives/shape_of.hpp"

namespace ov::intel_gpu {

static void CreateShapeOfOpCommon(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    validate_inputs_count(op, {1, 2});
    const auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);

    const auto primitive = cldnn::shape_of(layerName,
                                     inputs[0],
                                     cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    p.add_primitive(*op, primitive);
}

static void CreateShapeOfOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::ShapeOf>& op) {
    CreateShapeOfOpCommon(p, op);
}

static void CreateShapeOfOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ShapeOf>& op) {
   CreateShapeOfOpCommon(p, op);
}

REGISTER_FACTORY_IMPL(v0, ShapeOf);
REGISTER_FACTORY_IMPL(v3, ShapeOf);

}  // namespace ov::intel_gpu
