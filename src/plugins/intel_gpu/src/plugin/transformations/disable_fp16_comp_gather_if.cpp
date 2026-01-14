// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_gather_if.hpp"

#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/if.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForDetectron2MaskRCNNGatherIfPattern::DisableFP16CompForDetectron2MaskRCNNGatherIfPattern() {

    auto root_pattern = ov::pass::pattern::wrap_type<ov::op::v8::If, ov::op::v8::Gather>();
    auto m = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "DisableFP16CompForDetectron2MaskRCNNGatherIfPattern");

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto matched_node = m.get_match_root();

        if (matched_node) {
            // check gather_nd -> if pattern
            auto if_op = std::dynamic_pointer_cast<ov::op::v8::If>(matched_node);
            if (if_op) {
                for (auto& input : if_op->inputs()) {
                    auto src_node = input.get_source_output().get_node_shared_ptr();

                    auto gather_nd = std::dynamic_pointer_cast<ov::op::v8::GatherND>(src_node);
                    if (!gather_nd || !gather_nd->get_element_type().is_real())
                        continue;

                    disable_fp16_compression(if_op);
                    return true;
                }
            }   

            // check gather -> gather pattern
            auto gather_op = std::dynamic_pointer_cast<ov::op::v8::Gather>(matched_node);
            if (gather_op) {
                for (auto& input : gather_op->inputs()) {
                    auto src_node = input.get_source_output().get_node_shared_ptr();

                    auto gather = std::dynamic_pointer_cast<ov::op::v8::Gather>(src_node);
                    if (!gather || !gather->get_element_type().is_real())
                        continue;

                    disable_fp16_compression(gather_op);
                    return true;
                }
            }   
        }

        return false;
    };

    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
