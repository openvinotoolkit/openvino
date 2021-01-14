// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/cum_sum.hpp"
#include "ngraph/op/constant.hpp"

#include "api/cum_sum.hpp"

namespace CLDNNPlugin {

static inline cldnn::cum_sum::cum_sum_axis GetCumSumAxis(int32_t axis, uint32_t rank) {
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        THROW_IE_EXCEPTION << "CumSum axis is not correspond to number of dimensions";

    // Difference in dimension ordering between IE and clDNN,
    // reverse spatial dimensions after batch and feature.
    uint32_t cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max(rank, 4u) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return cldnn::cum_sum::cum_sum_axis::along_b;
        case 1: return cldnn::cum_sum::cum_sum_axis::along_f;
        case 2: return cldnn::cum_sum::cum_sum_axis::along_x;
        case 3: return cldnn::cum_sum::cum_sum_axis::along_y;
        case 4: return cldnn::cum_sum::cum_sum_axis::along_z;
        case 5: return cldnn::cum_sum::cum_sum_axis::along_w;
        default: THROW_IE_EXCEPTION << "Unsupported CumSum axis: " << axis;
    }

    return cldnn::cum_sum::cum_sum_axis::along_f;  // shouldn't get here
}

void CreateCumSumOp(Program& p, const std::shared_ptr<ngraph::op::v0::CumSum>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto exclusive = op->is_exclusive();
    auto reverse = op->is_reverse();

    size_t rank = op->get_input_shape(0).size();
    int32_t axis = 0;
    if (op->get_input_size() == 2) {
        auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!axes_constant) {
            THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        axis = axes_constant->cast_vector<int32_t>()[0];
    }

    auto primitive = cldnn::cum_sum(layerName,
                                    inputPrimitives[0],
                                    GetCumSumAxis(axis, rank),
                                    exclusive,
                                    reverse);

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, CumSum);

}  // namespace CLDNNPlugin
