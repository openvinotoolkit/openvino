// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using MemoryTestParams = std::tuple<
        ngraph::helpers::MemoryTransformation,   // Apply Memory transformation
        int64_t,                            // iterationCount
        InferenceEngine::SizeVector,        // inputShape
        InferenceEngine::Precision,         // netPrecision
        std::string                         // targetDevice
>;

class MemoryTest : public testing::WithParamInterface<MemoryTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemoryTestParams> &obj);
    void Run() override;
protected:
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override;
    void SetUp() override;
private:
    void CreateTIFunc();
    void CreateCommonFunc();
    void ApplyLowLatency();

    InferenceEngine::Precision netPrecision;
    ngraph::EvaluationContext eval_context;
    ngraph::helpers::MemoryTransformation transformation;

    int64_t iteration_count;
    ngraph::element::Type ngPrc;
    InferenceEngine::SizeVector inputShape;
};

}  // namespace LayerTestsDefinitions
