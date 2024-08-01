// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/shuffle_channels.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/shuffle_channels.hpp"

namespace ov {
namespace intel_gpu {

static void CreateShuffleChannelsOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::ShuffleChannels>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t group = op->get_group();
    int64_t axis = ov::util::try_normalize_axis(op->get_axis(), op->get_input_partial_shape(0).rank(), *op);

    auto shuffleChannelsPrim = cldnn::shuffle_channels(layerName,
                                                       inputs[0],
                                                       group,
                                                       axis);

    p.add_primitive(*op, shuffleChannelsPrim);
}

REGISTER_FACTORY_IMPL(v0, ShuffleChannels);

}  // namespace intel_gpu
}  // namespace ov
