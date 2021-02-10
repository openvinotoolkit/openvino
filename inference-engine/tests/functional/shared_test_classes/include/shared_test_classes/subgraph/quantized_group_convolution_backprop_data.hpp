// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        std::vector<ptrdiff_t>,
        std::vector<ptrdiff_t>,
        InferenceEngine::SizeVector,
        size_t,
        size_t,
        ngraph::op::PadType,
        size_t,
        ngraph::helpers::QuantizationGranularity> quantGroupConvBackpropDataSpecificParams;
typedef std::tuple<
        quantGroupConvBackpropDataSpecificParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> quantGroupConvBackpropDataLayerTestParamsSet;

class QuantGroupConvBackpropDataLayerTest : public testing::WithParamInterface<quantGroupConvBackpropDataLayerTestParamsSet>,
                                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<quantGroupConvBackpropDataLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions