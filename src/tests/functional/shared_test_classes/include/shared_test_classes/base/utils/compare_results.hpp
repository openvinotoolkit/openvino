// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_core.hpp"
#include "ngraph/node.hpp"

namespace ov {
namespace test {
namespace utils {

using CompareMap = std::map<ov::NodeTypeInfo, std::function<void(
        const std::shared_ptr<ov::Node> &node,
        size_t port,
        const ov::runtime::Tensor &expected,
        const ov::runtime::Tensor &actual,
        double absThreshold,
        double relThreshold)>>;

CompareMap getCompareMap();

} // namespace utils
} // namespace test
} // namespace ov
