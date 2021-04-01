// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/shuffle_channels.hpp"

#include "api/shuffle_channels.hpp"

namespace CLDNNPlugin {

void CreateShuffleChannelsOp(Program& p, const std::shared_ptr<ngraph::op::v0::ShuffleChannels>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto in_rank = op->get_input_shape(0).size();

    int32_t group = op->get_group();
    int32_t axis = op->get_axis();

    if (axis < 0)
        axis += in_rank;

    if (axis < 0 || axis >= in_rank)
        THROW_IE_EXCEPTION << "Incorrect axis value! Actual axis is" + std::to_string(group);

    if (group < 1)
        THROW_IE_EXCEPTION << "Invalid group size value (should equal at least one). Actual block size is" << std::to_string(group);

    if (op->get_input_shape(0)[axis] % group != 0)
        THROW_IE_EXCEPTION << "Group parameter must evenly divide the channel dimension. Actual group size is " << std::to_string(axis);

    auto shuffleChannelsPrim = cldnn::shuffle_channels(layerName,
                                                       inputPrimitives[0],
                                                       group,
                                                       axis);

    p.AddPrimitive(shuffleChannelsPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ShuffleChannels);

}  // namespace CLDNNPlugin
