// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather_elements.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/gather_elements.hpp"

namespace ov {
namespace intel_gpu {

static void CreateGatherElementsOp(Program& p, const std::shared_ptr<ngraph::op::v6::GatherElements>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outLayout = cldnn::format::get_default_format(op->get_output_shape(0).size());

    size_t rank = op->get_input_shape(0).size();
    int64_t axis = op->get_axis();
    if (axis < 0)
        axis += rank;
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(rank),
                    "GatherElements axis is not correspond to number of dimensions");

    auto primitive = cldnn::gather_elements(layerName,
                                            inputPrimitives[0],
                                            inputPrimitives[1],
                                            outLayout,
                                            tensor_from_dims(op->get_output_shape(0)),
                                            axis,
                                            op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v6, GatherElements);

}  // namespace intel_gpu
}  // namespace ov
