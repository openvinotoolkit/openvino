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

namespace LayerTestsDefinitions {

using RNNCellParams = typename std::tuple<
        bool,                              // using decompose to sub-ops transformation
        size_t,                            // batch
        size_t,                            // hidden size
        size_t,                            // input size
        std::vector<std::string>,          // activations
        float,                             // clip
        ngraph::helpers::InputLayerType,   // W input type (Constant or Parameter)
        ngraph::helpers::InputLayerType,   // R input type (Constant or Parameter)
        ngraph::helpers::InputLayerType,   // B input type (Constant or Parameter)
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class RNNCellTest : public testing::WithParamInterface<RNNCellParams >,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNCellParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
