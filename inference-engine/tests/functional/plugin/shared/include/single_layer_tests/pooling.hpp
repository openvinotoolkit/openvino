// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ngraph::helpers::PoolingTypes,
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        ngraph::op::RoundingType,
        ngraph::op::PadType,
        bool> poolSpecificParams;
typedef std::tuple<
        poolSpecificParams,
        InferenceEngine::Precision,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        std::string> poolLayerTestParamsSet;

class PoolingLayerTest
        : public LayerTestsUtils::LayerTestsCommonClass<poolLayerTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<poolLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions