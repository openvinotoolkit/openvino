// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

typedef std::pair<float, float> QuantRange;

typedef std::tuple<
        size_t,
        QuantRange,
        QuantRange,
        ngraph::helpers::QuantizationGranularity,
        InferenceEngine::Precision> QuantParams;

typedef std::tuple<
        QuantParams,
        QuantParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> QuantMatMulLayerTestParamsSet;

class QuantMatMulTest : public testing::WithParamInterface<QuantMatMulLayerTestParamsSet>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantMatMulLayerTestParamsSet> &obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
