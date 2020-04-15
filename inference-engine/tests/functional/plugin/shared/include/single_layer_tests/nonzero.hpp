// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

using ConfigMap = typename std::map<std::string, std::string>;

using NonZeroLayerTestParamsSet = typename std::tuple<
        InferenceEngine::SizeVector,          // Input shapes
        InferenceEngine::Precision,           // Input precision
        InferenceEngine::Precision,           // Network precision
        std::string,                          // Device name
        ConfigMap>;                           // Config map

class NonZeroLayerTest
        : public LayerTestsUtils::LayerTestsCommonClass<NonZeroLayerTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NonZeroLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
