// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_core.hpp"
#include "ngraph/node.hpp"

#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
namespace utils {

using InputsMap = std::map<ov::NodeTypeInfo, std::function<ov::runtime::Tensor(
        const std::shared_ptr<ov::Node>& node,
        size_t port,
        const ov::element::Type& elemType,
        const ov::Shape& targetShape)>>;

InputsMap getInputMap();

} // namespace utils
} // namespace test
} // namespace ov
