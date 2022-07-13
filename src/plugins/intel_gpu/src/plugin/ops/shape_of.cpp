// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/shape_of.hpp"

#include "intel_gpu/primitives/shape_of.hpp"

namespace ov {
namespace intel_gpu {

static void CreateShapeOfOpCommon(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::shape_of(layerName,
                                     inputPrimitives[0],
                                     op->get_output_partial_shape(0).rank().get_length(),
                                     DataTypeFromPrecision(op->get_output_element_type(0)),
                                     op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

static void CreateShapeOfOp(Program& p, const std::shared_ptr<ngraph::op::v0::ShapeOf>& op) {
    CreateShapeOfOpCommon(p, op);
}

static void CreateShapeOfOp(Program& p, const std::shared_ptr<ngraph::op::v3::ShapeOf>& op) {
   CreateShapeOfOpCommon(p, op);
}

REGISTER_FACTORY_IMPL(v0, ShapeOf);
REGISTER_FACTORY_IMPL(v3, ShapeOf);

}  // namespace intel_gpu
}  // namespace ov
