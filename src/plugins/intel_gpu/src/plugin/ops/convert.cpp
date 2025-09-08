// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"

#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

static void CreateConvertLikeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ConvertLike>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = cldnn::element_type_to_data_type(op->get_input_element_type(1));

    auto reorderPrim = cldnn::reorder(layerName,
                                      inputs[0],
                                      cldnn::format::any,
                                      outDataType);
    p.add_primitive(*op, reorderPrim);
}

static void CreateConvertOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Convert>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = cldnn::element_type_to_data_type(op->get_destination_type());

    auto reorderPrim = cldnn::reorder(layerName,
                                      inputs[0],
                                      cldnn::format::any,
                                      outDataType,
                                      std::vector<float>(),
                                      cldnn::reorder_mean_mode::subtract,
                                      cldnn::padding(),
                                      true);

    p.add_primitive(*op, reorderPrim);
}

REGISTER_FACTORY_IMPL(v0, Convert);
REGISTER_FACTORY_IMPL(v1, ConvertLike);

}  // namespace ov::intel_gpu
