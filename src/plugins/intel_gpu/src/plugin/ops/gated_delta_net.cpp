// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/gated_delta_net.hpp>
#include "openvino/op/gated_delta_net.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace op {
namespace internal {
using GatedDeltaNet = ov::op::GatedDeltaNet;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateGatedDeltaNetOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatedDeltaNet>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gated_delta_net gated_delta_net_prim(layerName, inputs);
    gated_delta_net_prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, gated_delta_net_prim);
}

REGISTER_FACTORY_IMPL(internal, GatedDeltaNet);

}  // namespace ov::intel_gpu
