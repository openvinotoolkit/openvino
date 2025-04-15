// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/gather_elements.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/gather_elements.hpp"

namespace ov::intel_gpu {

static void CreateGatherElementsOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::GatherElements>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    size_t rank = op->get_input_partial_shape(0).size();
    int64_t axis = op->get_axis();
    if (axis < 0)
        axis += rank;
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(rank),
                    "GatherElements axis is not correspond to number of dimensions");

    std::shared_ptr<cldnn::gather_elements> primitive = nullptr;
    if (op->get_output_partial_shape(0).is_dynamic() || p.use_new_shape_infer()) {
        primitive = std::make_shared<cldnn::gather_elements>(layerName, inputs[0], inputs[1], axis);
    } else {
        auto outLayout = cldnn::format::get_default_format(op->get_output_shape(0).size());
        primitive = std::make_shared<cldnn::gather_elements>(layerName,
                                                             inputs[0],
                                                             inputs[1],
                                                             outLayout,
                                                             tensor_from_dims(op->get_output_shape(0)),
                                                             axis);
    }

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v6, GatherElements);

}  // namespace ov::intel_gpu
