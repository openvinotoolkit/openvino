// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/grn.hpp"

#include "intel_gpu/primitives/grn.hpp"

namespace ov::intel_gpu {

static void CreateGRNOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::GRN>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::grn(layerName,
                                inputs[0],
                                op->get_bias(),
                                cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v0, GRN);

}  // namespace ov::intel_gpu
