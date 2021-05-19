// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_core.hpp>
#include "ngraph/node.hpp"

namespace LayerTestsDefinitions {

using CompareMap = std::map<ngraph::NodeTypeInfo, std::function<void(
        const std::shared_ptr<ngraph::Node> node,
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
        const std::vector<InferenceEngine::Blob::Ptr>& actual,
        float threshold)>>;

CompareMap getCompareMap();
} // namespace LayerTestsDefinitions
