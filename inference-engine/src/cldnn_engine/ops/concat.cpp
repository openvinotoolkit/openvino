// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/concat.hpp"

#include "api/concatenation.hpp"

namespace CLDNNPlugin {

static cldnn::concatenation::concatenation_axis GetConcatAxis(int32_t axis, size_t rank) {
    if (axis >= rank)
        THROW_IE_EXCEPTION << "Concatenation axis exceeds number of dimensions";

    // Difference in dimension ordering between IE and clDNN,
    // reverse spatial dimensions after batch and feature.
    unsigned cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return cldnn::concatenation::concatenation_axis::along_b;
        case 1: return cldnn::concatenation::concatenation_axis::along_f;
        case 2: return cldnn::concatenation::concatenation_axis::along_x;
        case 3: return cldnn::concatenation::concatenation_axis::along_y;
        case 4: return cldnn::concatenation::concatenation_axis::along_z;
        case 5: return cldnn::concatenation::concatenation_axis::along_w;
        default: THROW_IE_EXCEPTION << "Unsupported concatenation axis: " << axis;
    }

    return cldnn::concatenation::concatenation_axis::along_f;  // shouldn't get here
}

void CreateConcatOp(Program& p, const std::shared_ptr<ngraph::op::v0::Concat>& op) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto concatPrim = cldnn::concatenation(
        layerName,
        inputPrimitives,
        GetConcatAxis(op->get_axis(), op->get_input_shape(0).size()),
        DataTypeFromPrecision(op->get_output_element_type(0)));

    p.AddPrimitive(concatPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, Concat);

}  // namespace CLDNNPlugin
