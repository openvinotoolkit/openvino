// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov::intel_gpu {

static void CreateIdentityOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::Identity>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> order;
    int rank = std::max(4, static_cast<int>(op->get_input_partial_shape(0).size()));

    // if order size is less than 4 - fill the rest with just copy
    for (int o = rank - 1; o >= 0; o--) {
        order.push_back((uint16_t)o);
    }

    auto permutePrim = cldnn::permute(layerName, inputs[0], order);
    permutePrim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, permutePrim);
}

REGISTER_FACTORY_IMPL(v16, Identity);

}  // namespace ov::intel_gpu
