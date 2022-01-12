// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using FuseTransposeAndReorderParams = std::tuple<
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::Precision      // Input precision
>;

class FuseTransposeAndReorderTest : public testing::WithParamInterface<FuseTransposeAndReorderParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj);

protected:
    void SetUp() override;
    virtual void CreateGraph();
    void CheckTransposeCount(size_t expectedTransposeCount);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision inPrec;
};

class FuseTransposeAndReorderTest1 : public FuseTransposeAndReorderTest {
protected:
    void CreateGraph() override;
};

class FuseTransposeAndReorderTest2 : public FuseTransposeAndReorderTest {
protected:
    void CreateGraph() override;
};

} // namespace SubgraphTestsDefinitions
