// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/check_node_type.hpp"

#include <gtest/gtest.h>

#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace test {
namespace utils {

inline std::string setToString(const std::unordered_set<std::string> s) {
    return std::string("{") + ov::util::join(s) + "}";
}

void CheckNumberOfNodesWithTypeImpl(std::shared_ptr<const ov::Model> function,
                                    const std::unordered_set<std::string>& nodeTypes,
                                    size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    size_t actualNodeCount = 0;
    for (const auto& node : function->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        if (nodeTypes.count(getExecValue(ov::exec_model_info::LAYER_TYPE))) {
            actualNodeCount++;
        }
    }

    ASSERT_EQ(expectedCount, actualNodeCount)
        << "Unexpected count of the node types '" << setToString(nodeTypes) << "' ";
}
}  // namespace utils
}  // namespace test
}  // namespace ov