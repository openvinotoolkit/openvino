// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_parallel.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include <memory>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/sync_tensor.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

TensorParallelFusion::TensorParallelFusion(size_t world_size) {
    auto fully_connected_m = ov::pass::pattern::wrap_type<ov::intel_gpu::op::FullyConnected>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();

        const auto& fc = std::dynamic_pointer_cast<ov::intel_gpu::op::FullyConnected>(m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }
        // ignore compressed for now
        if (std::dynamic_pointer_cast<op::FullyConnectedCompressed>(fc))
            return false;
        std::map<int, std::shared_ptr<ov::Node>> org_users;
        for (auto u : fc->get_users()) {
            for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                if (u->get_input_node_shared_ptr(idx) == fc) {
                    org_users.insert({idx, u});
                }
            }
        }
        auto wgt_item = fc->get_input_node_shared_ptr(1);
        // split weight
        auto split_dim_range = wgt_item->get_shape()[0];
        auto sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(fc, world_size, split_dim_range, fc->get_element_type());
        sync_node->set_friendly_name(fc->get_friendly_name()+ "_TP");

        auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
        concat_node->set_friendly_name(fc->get_friendly_name()+ "_ALLGATHER");
        copy_runtime_info(fc, concat_node);
        for (auto& iter : org_users) {
            iter.second->input(iter.first).replace_source_output(concat_node->output(0));
        }
        fc->clear_control_dependencies();
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "TensorParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov