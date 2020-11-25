// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

using FusePermuteAndReorderParams = std::tuple<
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::Precision      // Input precision
>;

class FusePermuteAndReorderTest : public testing::WithParamInterface<FusePermuteAndReorderParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FusePermuteAndReorderParams> obj);

protected:
    void SetUp() override;
    virtual void CreateGraph();
    void CheckPermuteCount(size_t expectedPermuteCount);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision inPrec;
};

class FusePermuteAndReorderTest1 : public FusePermuteAndReorderTest {
protected:
    void CreateGraph() override;
};

class FusePermuteAndReorderTest2 : public FusePermuteAndReorderTest {
protected:
    void CreateGraph() override;
};

} // namespace LayerTestsDefinitions
