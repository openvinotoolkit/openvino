// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov::intel_gpu {

static void CreateTransposeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Transpose>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> order;
    if (op->get_input_size() == 2) {
        auto order_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(order_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        order = order_constant->cast_vector<uint16_t>();
    }

    int rank = std::max(4, static_cast<int>(op->get_input_partial_shape(0).size()));
    if (order.empty()) {
        // if order size is less than 4 - fill the rest with just copy
        for (int o = rank - 1; o >= 0; o--)
            order.push_back((uint16_t)o);
    }

    auto permutePrim = cldnn::permute(layerName,
                                      inputs[0],
                                      order);
    permutePrim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, permutePrim);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace ov::intel_gpu
