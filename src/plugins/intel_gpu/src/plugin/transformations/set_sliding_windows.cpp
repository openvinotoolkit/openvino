// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_sliding_windows.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/core/graph_util.hpp"
#include <memory>

namespace ov::intel_gpu {

bool SetSlidingWindows::run_on_model(const std::shared_ptr<ov::Model>& m) {
    size_t idx = 0;
    for (const auto& op : m->get_ops()) {
        if (auto paged_attn_op = ov::as_type_ptr<ov::op::PagedAttentionExtension>(op)) {
            auto sliding_windows_const_orig = ov::as_type_ptr<ov::op::v0::Constant>(paged_attn_op->get_input_node_shared_ptr(10));
            auto new_sliding_windows_const = std::make_shared<ov::op::v0::Constant>(sliding_windows_const_orig->get_output_element_type(0),
                                                                                    ov::Shape{},
                                                                                    sliding_window_per_layer[idx++]);
            ov::replace_node(sliding_windows_const_orig, new_sliding_windows_const);
        }
    }
    return false;
}

}  // namespace ov::intel_gpu
