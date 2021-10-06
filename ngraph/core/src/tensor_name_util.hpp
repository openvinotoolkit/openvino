// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/function.hpp"

namespace ov {

/// \brief Check that specified tensor name is unique for a given function.
///
/// \param tensor_name Name to check across all tensors in a function.
/// \param function Function.
/// \return False if tensor name is already used in some function's node, True otherwise
inline bool is_tensor_name_available(const std::string& tensor_name, const std::shared_ptr<Function>& function) {
    for (const auto& node : function->get_ordered_ops()) {
        for (const auto& output : node->outputs()) {
            const auto& tensor = output.get_tensor();
            if (tensor.get_names().count(tensor_name)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace ov
