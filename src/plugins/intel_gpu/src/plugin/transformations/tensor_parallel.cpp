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
#include "openvino/op/add.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

TensorParallelFusion::TensorParallelFusion() {
    auto fully_connected_m = ov::pass::pattern::wrap_type<ov::intel_gpu::op::FullyConnected>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();

        const auto& fc = std::dynamic_pointer_cast<ov::intel_gpu::op::FullyConnected>(m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }
        // ignore compressed now
        if (std::dynamic_pointer_cast<op::FullyConnectedCompressed>(fc))
            return false;
        auto sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(fc, fc->get_element_type());
        sync_node->set_friendly_name(fc->get_friendly_name()+ "_TP");

        copy_runtime_info(fc, sync_node);
        ov::replace_node(fc, sync_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "TensorParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov