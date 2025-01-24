// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/reverse_sequence.hpp"

#include "intel_gpu/primitives/reverse_sequence.hpp"

namespace ov::intel_gpu {

static void CreateReverseSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::ReverseSequence>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto batch_axis = static_cast<uint32_t>(op->get_batch_axis());
    auto seq_axis = static_cast<uint32_t>(op->get_sequence_axis());
    auto reverseSequencePrim = cldnn::reverse_sequence(layerName,
                                                       inputs[0],
                                                       inputs[1],
                                                       seq_axis,
                                                       batch_axis);

    p.add_primitive(*op, reverseSequencePrim);
}

REGISTER_FACTORY_IMPL(v0, ReverseSequence);

}  // namespace ov::intel_gpu
