// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather_elements.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/gather_elements.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static cldnn::gather_elements::gather_elements_axis GetGatherAxis(int axis, unsigned rank) {
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        IE_THROW() << "GatherElements axis is not correspond to number of dimensions";

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
        case 0: return cldnn::gather_elements::gather_elements_axis::along_b;
        case 1: return cldnn::gather_elements::gather_elements_axis::along_f;
        case 2: return cldnn::gather_elements::gather_elements_axis::along_x;
        case 3: return cldnn::gather_elements::gather_elements_axis::along_y;
        case 4: return cldnn::gather_elements::gather_elements_axis::along_z;
        case 5: return cldnn::gather_elements::gather_elements_axis::along_w;
        default: IE_THROW() << "Unsupported GatherElements axis: " << axis;
    }
    return cldnn::gather_elements::gather_elements_axis::along_f;  // shouldn't get here
}

static void CreateGatherElementsOp(Program& p, const std::shared_ptr<ngraph::op::v6::GatherElements>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t rank = op->get_input_shape(0).size();
    int32_t axis = static_cast<int32_t>(op->get_axis());

    auto outLayout = DefaultFormatForDims(op->get_output_shape(0).size());

    auto primitive = cldnn::gather_elements(layerName,
                                            inputPrimitives[0],
                                            inputPrimitives[1],
                                            outLayout,
                                            tensor_from_dims(op->get_output_shape(0)),
                                            GetGatherAxis(axis, rank),
                                            op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v6, GatherElements);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
