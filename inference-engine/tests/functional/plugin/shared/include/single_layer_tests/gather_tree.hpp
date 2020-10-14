// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using GatherTreeParamsTuple = typename std::tuple<
        std::vector<size_t>,               // Input tensors shape
        ngraph::helpers::InputLayerType,   // Secondary input type
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string>;                      // Device name

class GatherTreeLayerTest : public testing::WithParamInterface<GatherTreeParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions