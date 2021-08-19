// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"


namespace SubgraphTestsDefinitions {

using FuseMulAddAndEwSimpleParams = std::tuple<
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::Precision      // Input precision
>;

class FuseMulAddAndEwSimpleTest : public testing::WithParamInterface<FuseMulAddAndEwSimpleParams>, public CPUTestUtils::CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMulAddAndEwSimpleParams> obj);

protected:
    void SetUp() override;
    virtual void CreateGraph() = 0;

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision inPrec;
};

class FuseMulAddAndEwSimpleTest1 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

class FuseMulAddAndEwSimpleTest2 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

class FuseMulAddAndEwSimpleTest3 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

} // namespace SubgraphTestsDefinitions
