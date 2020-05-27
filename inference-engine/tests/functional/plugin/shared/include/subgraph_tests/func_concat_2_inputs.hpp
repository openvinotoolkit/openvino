// Copyright (C) 2020Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "functional_test_utils/layer_test_utils.hpp"


namespace LayerTestsDefinitions {

// config name and actual config - threshold per config
using ConfigTuple = std::tuple<std::string, float, float, std::map<std::string, std::string>>;

using NonTrivialConcatParams = decltype (std::tuple_cat(LayerTestsUtils::basicParams(), ConfigTuple()));

class NonTrivialConcat2Inputs : public testing::WithParamInterface<NonTrivialConcatParams>,
                        public LayerTestsUtils::LayerTestsCommon {
 public:
    static std::string getTestCaseName(testing::TestParamInfo<NonTrivialConcatParams> obj);
    float threshold(const InferenceEngine::Precision & precision) override {
        return fp32Threshold;
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 10, 0, 10);
    }

 protected:
    void SetUp() override;
    float fp32Threshold  = 0.0f;
    float fp32InputRange = 0.0f;
};

}  // namespace LayerTestsDefinitions