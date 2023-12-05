// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,                 // Input Shapes
        std::vector<size_t>,                 // Kernel Shape
        size_t                               // Stride
> convParams;

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        convParams,                          // Convolution Params
        size_t,                              // Input Channels
        size_t                               // Output Channels
> EltwiseConvEltwiseParams;

class EltwiseAfterConvTest : public testing::WithParamInterface<EltwiseConvEltwiseParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseConvEltwiseParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

class EltwiseBeforeConvTest : public testing::WithParamInterface<EltwiseConvEltwiseParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseConvEltwiseParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

class EltwiseWithTwoConvsAsInputsTest : public testing::WithParamInterface<EltwiseConvEltwiseParams>,
                                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseConvEltwiseParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
