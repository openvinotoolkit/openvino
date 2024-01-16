// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<bool,                               // SpecialZero
                   InferenceEngine::Precision,         // Network precision
                   InferenceEngine::Precision,         // Input precision
                   InferenceEngine::Precision,         // Output precision
                   InferenceEngine::Layout,            // Input layout
                   InferenceEngine::Layout,            // Output layout
                   std::vector<size_t>,                // Input shapes
                   std::vector<int64_t>,               // OutForm Shapes
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
    reshapeParams;
class ReshapeLayerTest : public testing::WithParamInterface<reshapeParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reshapeParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
