// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/stft.hpp"

namespace ov::intel_gpu {

static void CreateSTFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::STFT>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    auto prim =
        cldnn::STFT(layer_type_name_ID(op), inputs[0], inputs[1], inputs[2], inputs[3], op->get_transpose_frames());
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v15, STFT);

}  // namespace ov::intel_gpu
