// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/reverse_sequence.hpp"

#include "api/reverse_sequence.hpp"

namespace CLDNNPlugin {

void CreateReverseSequenceOp(Program& p, const std::shared_ptr<ngraph::op::v0::ReverseSequence>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t batch_axis = op->get_batch_axis();
    size_t seq_axis = op->get_sequence_axis();
    auto reverseSequencePrim = cldnn::reverse_sequence(layerName,
                                                       inputPrimitives[0],
                                                       inputPrimitives[1],
                                                       seq_axis,
                                                       batch_axis);

    p.AddPrimitive(reverseSequencePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ReverseSequence);

}  // namespace CLDNNPlugin
