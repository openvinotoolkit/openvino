// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/remove_rt_info.hpp"

#include "openvino/cc/pass/itt.hpp"

bool ov::pass::RemoveRtInfo::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(RemoveRtInfo);

    for (auto& node : f->get_ops()) {
        node->get_rt_info().clear();
    }
    return false;
}
