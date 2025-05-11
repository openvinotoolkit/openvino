// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"

namespace ov {

inline std::unordered_set<std::string> get_function_tensor_names(const std::shared_ptr<Model>& function) {
    std::unordered_set<std::string> set;
    for (const auto& node : function->get_ordered_ops()) {
        for (const auto& output : node->outputs()) {
            const auto& names = output.get_tensor().get_names();
            set.insert(names.begin(), names.end());
        }
    }
    return set;
}

}  // namespace ov
