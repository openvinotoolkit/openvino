// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/op/sin.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/scatter_base.hpp"
#include "openvino/op/util/scatter_nd_base.hpp"
#include "openvino/op/util/scatter_elements_update_base.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/gather_nd_base.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "transformations/utils/utils.hpp"
#include "disable_fp16_comp_for_periodic_funcs.hpp"

static bool is_non_value_modifying_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Reshape>(node) ||
           ov::is_type<ov::op::v1::Transpose>(node) ||
           ov::is_type<ov::op::util::ScatterBase>(node) ||
           ov::is_type<ov::op::util::ScatterNDBase>(node) ||
           ov::is_type<ov::op::util::ScatterElementsUpdateBase>(node) ||
           ov::is_type<ov::op::v8::Slice>(node) ||
           ov::is_type<ov::op::v1::Broadcast>(node) ||
           ov::is_type<ov::op::v0::Concat>(node) ||
           ov::is_type<ov::op::v1::Split>(node) ||
           ov::is_type<ov::op::v1::StridedSlice>(node) ||
           ov::is_type<ov::op::v0::Tile>(node) ||
           ov::is_type<ov::op::v16::Identity>(node) ||
           ov::is_type<ov::op::v1::Pad>(node) ||
           ov::is_type<ov::op::util::GatherNDBase>(node) ||
           ov::is_type<ov::op::util::GatherBase>(node) ||
           ov::is_type<ov::op::v15::Squeeze>(node) ||
           ov::is_type<ov::op::v0::Unsqueeze>(node);
}

ov::intel_gpu::DisableFP16CompressionForPeriodicFuncs::DisableFP16CompressionForPeriodicFuncs()
    : ov::pass::MatcherPass() {
    set_name("DisableFP16CompressionForPeriodicFuncs");

    auto cos = ov::pass::pattern::wrap_type<ov::op::v0::Cos>();
    auto sin = ov::pass::pattern::wrap_type<ov::op::v0::Sin>();
    auto sin_cos_func_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cos, sin});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        // Disable FP16 compression for the current node
        ov::disable_fp16_compression(node);

        auto current_node = node;

        OPENVINO_ASSERT(current_node && !current_node->inputs().empty(),
                        "current_node should not be null and have inputs");

        // Traverse the input chain to find the first value-modifying node
        while (current_node) {
            /*
            * Move to the first input node, the reason that we only traverse first input:
            * 1. Sin/Cos operations have only one input containing the actual data
            * 2. For non-value-modifying operations (reshape, transpose, etc.), input[0] contains the actual data
            *    while other inputs (input[1], input[2]...) are metadata for:
            *    - Shape information (target shape for reshape)
            *    - Index information (axes for transpose, gather indices)
            *    - Broadcasting parameters (broadcast shape)
            *    These metadata inputs don't affect the actual data values, only how the data is arranged or accessed
            * 3. Since we're tracking the data flow that affects numerical precision,
            *    we only need to follow the path of actual data (input[0])
            */
            current_node = current_node->get_input_node_shared_ptr(0);  // Follow only the first input
            if (!current_node || current_node->inputs().empty()) {
                break;
            }

            if (ov::fp16_compression_is_disabled(current_node)) {
                break;
            }

            if (!is_non_value_modifying_node(current_node)) {
                ov::disable_fp16_compression(current_node);
                break;
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin_cos_func_pattern, "DisableFP16CompressionForPeriodicFuncs");
    register_matcher(m, callback);
}
