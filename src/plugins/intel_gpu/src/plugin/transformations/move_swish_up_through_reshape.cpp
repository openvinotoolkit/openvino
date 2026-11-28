// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_swish_up_through_reshape.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

MoveSwishUpThroughReshape::MoveSwishUpThroughReshape() {
    MATCHER_SCOPE(MoveSwishUpThroughReshape);

    using namespace ov::pass;
    using namespace ov::pass::pattern;

    auto reshape_m = wrap_type<ov::op::v1::Reshape>(consumers_count(1));
    auto swish_m = wrap_type<ov::op::v4::Swish>({reshape_m, any_input()});
    auto swish_no_beta_m = wrap_type<ov::op::v4::Swish>({reshape_m});
    auto root_m = swish_m | swish_no_beta_m;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto swish = ov::as_type_ptr<ov::op::v4::Swish>(m.get_match_root());
        const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(reshape_m).get_node_shared_ptr());
        OPENVINO_ASSERT(swish, "MoveSwishUpThroughReshape: matched root is expected to be v4::Swish");
        OPENVINO_ASSERT(reshape, "MoveSwishUpThroughReshape: matched node is expected to be v1::Reshape");

        // consumers_count(1) already guarantees the Reshape feeds only this Swish, so no
        // other consumer can observe the pre-activation values.
        if (transformation_callback(swish)) {
            return false;
        }

        // Hoisting over a PagedCausalConv1D producer lands the Swish in the conv's OCL epilogue,
        // which reproduces the un-fused rounding exactly (round to f16, then SiLU in f16).
        //
        // Hoisting over a FullyConnected producer (the `in_proj_z` gate) instead turns the Swish into a
        // oneDNN eltwise_swish post-op evaluated on the f32 GEMM accumulator. That is more accurate than
        // the f16 standalone kernel but not bit-identical.

        // Rebuild as Reshape(Swish(x)). The Swish takes the Reshape's data input; its beta
        // input, when present, is a scalar and is carried over untouched.
        ov::OutputVector swish_inputs{reshape->input_value(0)};
        if (swish->get_input_size() > 1) {
            swish_inputs.push_back(swish->input_value(1));
        }
        auto new_swish = swish->clone_with_new_inputs(swish_inputs);
        new_swish->set_friendly_name(swish->get_friendly_name() + "_hoisted");

        auto new_reshape = reshape->clone_with_new_inputs({new_swish, reshape->input_value(1)});
        new_reshape->set_friendly_name(swish->get_friendly_name());

        ov::copy_runtime_info({reshape, swish}, {new_swish, new_reshape});
        ov::replace_node(swish, new_reshape);

        return true;
    };

    auto m = std::make_shared<Matcher>(root_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
