// Copyright (C) 2021 Intel Corporation
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

using ConvDWConvPatternParams =  std::tuple<
        size_t,                         // num out channels
        size_t,                         // dw stride
        bool,                           // with bias (1x1 conv)
        bool                            // with bias (dw conv)
>;

using ConvDWConvFusingParams = std::tuple<
        ConvDWConvPatternParams,        // Convolution params
        InferenceEngine::SizeVector     // Input shapes
>;

class ConvDWConvFusingSubgraphTest : public testing::WithParamInterface<ConvDWConvFusingParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvDWConvFusingParams> obj);

protected:
    void SetUp() override;
    void CheckConvCount();

    bool isFusedConvExpected(const ngraph::Shape& inShape, const ngraph::Shape& outShape, const size_t elemSize);
    size_t expectedConvCount;
};

} // namespace SubgraphTestsDefinitions
