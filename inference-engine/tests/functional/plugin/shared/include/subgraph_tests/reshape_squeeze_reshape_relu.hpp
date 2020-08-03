// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
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
} // namespace LayerTestsDefinitions
