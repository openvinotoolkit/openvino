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

using ReverseSequenceParamsTuple = typename std::tuple<
        int64_t,                           // Index of the batch dimension
        int64_t,                           // Index of the sequence dimension
        std::vector<size_t>,               // Input shapes
        std::vector<size_t>,               // Shape of the input vector with sequence lengths to be reversed
        ngraph::helpers::InputLayerType,   // Secondary input type
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class ReverseSequenceLayerTest : public testing::WithParamInterface<ReverseSequenceParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReverseSequenceParamsTuple> &obj);

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
