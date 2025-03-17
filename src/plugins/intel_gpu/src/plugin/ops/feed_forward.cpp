// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/feed_forward.hpp"
#include "intel_gpu/primitives/feed_forward.hpp"


namespace ov {
namespace op {
namespace internal {
using FeedForward = ov::intel_gpu::op::FeedForward;
}  // namespace internal
}  // namespace op
}  //

namespace ov::intel_gpu {

static void CreateFeedForwardOp(ProgramBuilder& p, const std::shared_ptr<op::FeedForward>& op) {
    validate_inputs_count(op, {5});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    if (p.use_new_shape_infer()) {
        auto prim = cldnn::feed_forward(primitive_name,
                                  inputs[0],
                                  inputs[1],
                                  inputs[2],
                                  inputs[3],
                                  inputs[4],
                                  cldnn::tensor());
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
    } else {
        auto prim = cldnn::feed_forward(primitive_name,
                                  inputs[0],
                                  inputs[1],
                                  inputs[2],
                                  inputs[3],
                                  inputs[4],
                                  tensor_from_dims(op->get_output_shape(0)));
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
    }
}

REGISTER_FACTORY_IMPL(internal, FeedForward);

}  // namespace ov::intel_gpu
