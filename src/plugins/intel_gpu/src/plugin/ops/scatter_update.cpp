// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/scatter_update.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/scatter_update.hpp"

namespace ov {
namespace intel_gpu {

static inline cldnn::scatter_update::scatter_update_axis GetScatterUpdateAxis(int axis, unsigned rank) {
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        IE_THROW() << "ScatterUpdate axis is not correspond to number of dimensions";

    // Difference in dimension ordering between IE and GPU plugin,
    // reverse spatial dimensions after batch and feature.
    unsigned cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max(rank, 4u) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return cldnn::scatter_update::scatter_update_axis::along_b;
        case 1: return cldnn::scatter_update::scatter_update_axis::along_f;
        case 2: return cldnn::scatter_update::scatter_update_axis::along_x;
        case 3: return cldnn::scatter_update::scatter_update_axis::along_y;
        case 4: return cldnn::scatter_update::scatter_update_axis::along_z;
        case 5: return cldnn::scatter_update::scatter_update_axis::along_w;
        default: IE_THROW() << "Unsupported ScatterUpdate axis: " << axis;
    }

    return cldnn::scatter_update::scatter_update_axis::along_f;  // shouldn't get here
}

static void CreateScatterUpdateOp(Program& p, const std::shared_ptr<ngraph::op::v3::ScatterUpdate>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t rank = op->get_input_shape(0).size();
    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(3));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    int32_t axis = axes_constant->cast_vector<int32_t>()[0];

    auto primitive = cldnn::scatter_update(layerName,
                                           inputPrimitives[0],
                                           inputPrimitives[1],
                                           inputPrimitives[2],
                                           GetScatterUpdateAxis(axis, rank),
                                           op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, ScatterUpdate);

}  // namespace intel_gpu
}  // namespace ov
