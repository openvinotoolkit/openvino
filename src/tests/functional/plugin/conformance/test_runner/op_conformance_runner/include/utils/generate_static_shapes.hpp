// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace utils {

using ShapesMap = std::map<ov::NodeTypeInfo, std::function<InputShape(
        const std::shared_ptr<ov::Node>& node,
        size_t in_port_id)>>;

ShapesMap getShapeMap();
} // namespace utils
} // namespace test
} // namespace ov
