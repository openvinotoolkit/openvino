// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/batch_to_space.hpp"

namespace ov {
namespace intel_gpu {

static void CreateBatchToSpaceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::BatchToSpace>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto rank = op->get_input_partial_shape(0).size();
    auto format = cldnn::format::get_default_format(rank);

    std::vector<std::vector<int32_t>> const_inputs;
    const_inputs.reserve(3);

    bool non_constant_input = false;
    for (size_t i = 1; i < 4; ++i) {
        auto inConst = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));

        bool is_const_input = (inConst != nullptr);
        OPENVINO_ASSERT((i == 1) || (i >= 2 && non_constant_input != is_const_input),
            "[GPU] Unsupported mixed node with constant and parameter in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        if (!inConst) {
            non_constant_input = true;
        }
    }

    if (non_constant_input) {
        auto batchToSpacePrim = cldnn::batch_to_space(layerName, inputs);
        p.add_primitive(*op, batchToSpacePrim);
    } else {
        for (size_t i = 1; i < 4; ++i) {
            auto in_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));
            const_inputs.emplace_back(in_const->cast_vector<int32_t>());
        }

        auto batchToSpacePrim = cldnn::batch_to_space(layerName,
                                                      inputs[0],           // input
                                                      const_inputs[0],     // block_shape
                                                      const_inputs[1],     // crops_begin
                                                      const_inputs[2]);    // crops_end


        p.add_primitive(*op, batchToSpacePrim);
    }
}

REGISTER_FACTORY_IMPL(v1, BatchToSpace);

}  // namespace intel_gpu
}  // namespace ov
