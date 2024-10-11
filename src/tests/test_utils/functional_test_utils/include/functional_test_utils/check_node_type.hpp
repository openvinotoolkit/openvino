// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/common_utils.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace test {
namespace utils {

void CheckNumberOfNodesWithTypeImpl(std::shared_ptr<const ov::Model> function,
                                    const std::unordered_set<std::string>& nodeTypes,
                                    size_t expectedCount);

inline void CheckNumberOfNodesWithTypes(const ov::CompiledModel& compiledModel,
                                        const std::unordered_set<std::string>& nodeTypes,
                                        size_t expectedCount) {
    if (!compiledModel)
        return;

    std::shared_ptr<const ov::Model> function = compiledModel.get_runtime_model();

    CheckNumberOfNodesWithTypeImpl(function, nodeTypes, expectedCount);
}

inline void CheckNumberOfNodesWithType(const ov::CompiledModel& compiledModel,
                                       const std::string& nodeType,
                                       size_t expectedCount) {
    CheckNumberOfNodesWithTypes(compiledModel, {nodeType}, expectedCount);
}

}  // namespace utils
}  // namespace test
}  // namespace ov
