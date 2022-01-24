// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_core.hpp>
#include "ngraph/node.hpp"

namespace LayerTestsDefinitions {
using InputsMap = std::map<ngraph::NodeTypeInfo, std::function<InferenceEngine::Blob::Ptr(
        const std::shared_ptr<ngraph::Node> node,
        const InferenceEngine::InputInfo& info,
        size_t port)>>;

InputsMap getInputMap();
} // namespace LayerTestsDefinitions
