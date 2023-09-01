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

using concatParamsTuple = typename std::tuple<
        int,                               // Concat axis
        std::vector<std::vector<size_t>>,  // Input shapes
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string>;                      // Device name

// Multichannel
class ConcatLayerTest : public testing::WithParamInterface<concatParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
