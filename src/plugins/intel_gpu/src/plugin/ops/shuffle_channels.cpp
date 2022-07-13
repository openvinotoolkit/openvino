// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/shuffle_channels.hpp"

#include "intel_gpu/primitives/shuffle_channels.hpp"

namespace ov {
namespace intel_gpu {

static void CreateShuffleChannelsOp(Program& p, const std::shared_ptr<ngraph::op::v0::ShuffleChannels>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto in_rank = op->get_input_shape(0).size();

    int32_t group = op->get_group();
    int32_t axis = op->get_axis();

    if (axis < 0)
        axis += in_rank;

    if (axis < 0 || axis >= in_rank)
        IE_THROW() << "Incorrect axis value! Actual axis is" + std::to_string(group);

    if (group < 1)
        IE_THROW() << "Invalid group size value (should equal at least one). Actual block size is" << std::to_string(group);

    if (op->get_input_shape(0)[axis] % group != 0)
        IE_THROW() << "Group parameter must evenly divide the channel dimension. Actual group size is " << std::to_string(axis);

    auto shuffleChannelsPrim = cldnn::shuffle_channels(layerName,
                                                       inputPrimitives[0],
                                                       group,
                                                       axis,
                                                       op->get_friendly_name());

    p.AddPrimitive(shuffleChannelsPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ShuffleChannels);

}  // namespace intel_gpu
}  // namespace ov
