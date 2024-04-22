// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/preprocess/color_format.hpp"

#include "shared_test_classes/base/utils/ranges.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"


namespace ov {
namespace test {
namespace utils {

using ShapesMap = std::map<ov::NodeTypeInfo, std::function<InputShape(
        const std::shared_ptr<ov::Node>& node,
        const std::shared_ptr<ov::op::v0::Parameter>& param)>>;

ShapesMap getShapeMap();
} // namespace utils
} // namespace test
} // namespace ov