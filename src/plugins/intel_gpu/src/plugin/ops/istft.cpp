// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/istft.hpp"

namespace ov::intel_gpu {

static void CreateISTFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::ISTFT>& op) {
    validate_inputs_count(op, {4, 5});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::ISTFT(layer_type_name_ID(op), inputs, op->get_center(), op->get_normalized());
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v16, ISTFT);

}  // namespace ov::intel_gpu
