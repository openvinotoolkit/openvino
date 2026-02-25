// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fused_names_cleanup.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::FusedNamesCleanup::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(FusedNamesCleanup);

    for (auto& node : f->get_ordered_ops()) {
        ov::op::util::process_subgraph(*this, node);

        RTMap& rt_info = node->get_rt_info();
        auto it = rt_info.find(ov::FusedNames::get_type_info_static());
        if (it != rt_info.end()) {
            rt_info.erase(it);
        }
    }
    return false;
}
