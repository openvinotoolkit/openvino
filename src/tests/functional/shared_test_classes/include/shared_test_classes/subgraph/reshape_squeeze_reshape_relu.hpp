// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {
using ShapeAxesTuple = std::pair<std::vector<size_t>, std::vector<int>>;

using ReshapeSqueezeReshapeReluTuple = typename std::tuple<
    ShapeAxesTuple,                     // Input shapes & squeeze_indices
    InferenceEngine::Precision,       // Network precision
    std::string,                      // Device name
    ngraph::helpers::SqueezeOpType    // SqueezeOpType
>;

class ReshapeSqueezeReshapeRelu
        : public testing::WithParamInterface<ReshapeSqueezeReshapeReluTuple>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
