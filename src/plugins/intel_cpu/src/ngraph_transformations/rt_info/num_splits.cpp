// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "num_splits.hpp"
#include <openvino/core/node.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
bool has_num_splits(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(NumSplits::get_type_info_static());
}
size_t get_num_splits(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().at(NumSplits::get_type_info_static()).as<NumSplits>().get_value();
}
void set_num_splits(const std::shared_ptr<ov::Node>& node, const size_t num_splits) {
    node->get_rt_info().emplace(NumSplits::get_type_info_static(), NumSplits{num_splits});
}
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
