// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/shuffle_channels.hpp"

#include "intel_gpu/primitives/shuffle_channels.hpp"

namespace ov {
namespace intel_gpu {

static void CreateShuffleChannelsOp(Program& p, const std::shared_ptr<ngraph::op::v0::ShuffleChannels>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t group = op->get_group();
    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = ov::normalize_axis(op.get(), op->get_axis(), op->get_input_partial_shape(0).rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto shuffleChannelsPrim = cldnn::shuffle_channels(layerName,
                                                       inputs[0],
                                                       group,
                                                       axis);

    p.add_primitive(*op, shuffleChannelsPrim);
}

REGISTER_FACTORY_IMPL(v0, ShuffleChannels);

}  // namespace intel_gpu
}  // namespace ov
