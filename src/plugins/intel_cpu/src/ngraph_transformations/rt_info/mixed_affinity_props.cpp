// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include "mixed_affinity_props.hpp"

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {

bool has_properties(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(MixedAffinityProps::get_type_info_static());
}

Properties get_properties(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().at(MixedAffinityProps::get_type_info_static()).as<MixedAffinityProps>().get_value();
}

void set_properties(const std::shared_ptr<ov::Node>& node, const Properties value) {
    node->get_rt_info().emplace(MixedAffinityProps::get_type_info_static(), MixedAffinityProps{value});
}

bool MixedAffinityProps::operator==(const MixedAffinityProps &rhs) const {
    return value.opt_bs == rhs.value.opt_bs && value.n_splits == rhs.value.n_splits;
}

bool MixedAffinityProps::operator!=(const MixedAffinityProps &rhs) const {
    return !(*this == rhs);
}

std::string MixedAffinityProps::to_string() const {
    std::string res = "bs=" + std::to_string(value.opt_bs) + ",n_splits=" + std::to_string(value.n_splits);
    return res;
}
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
