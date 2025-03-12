// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/lrn.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/lrn.hpp"

namespace ov::intel_gpu {

static cldnn::lrn_norm_region GetNormRegion(std::vector<int64_t> axis_value) {
    if (axis_value.size() == 1 && axis_value[0] == 1) {
        return cldnn::lrn_norm_region_across_channel;
    } else {
        return cldnn::lrn_norm_region_within_channel;
    }
}

static void CreateLRNOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::LRN>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(axis_const != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    auto axis_value = axis_const->cast_vector<int64_t>();
    auto localSize = static_cast<uint32_t>(op->get_nsize());

    auto lrnPrim = cldnn::lrn(layerName,
                              inputs[0],
                              localSize,
                              static_cast<float>(op->get_bias()),
                              static_cast<float>(op->get_alpha()),
                              static_cast<float>(op->get_beta()),
                              GetNormRegion(axis_value));

    p.add_primitive(*op, lrnPrim);
}

REGISTER_FACTORY_IMPL(v0, LRN);

}  // namespace ov::intel_gpu
