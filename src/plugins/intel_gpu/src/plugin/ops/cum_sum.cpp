// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/cum_sum.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"

namespace ov::intel_gpu {

static void CreateCumSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::CumSum>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto exclusive = op->is_exclusive();
    auto reverse = op->is_reverse();

    int64_t axis = 0;
    if (op->get_input_size() == 2) {
        auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        axis = axes_constant->cast_vector<int64_t>()[0];
    }
    axis = ov::util::try_normalize_axis(axis, op->get_input_partial_shape(0).rank(), *op);

    auto primitive = cldnn::cum_sum(layerName,
                                    inputs[0],
                                    axis,
                                    exclusive,
                                    reverse);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v0, CumSum);

}  // namespace ov::intel_gpu
