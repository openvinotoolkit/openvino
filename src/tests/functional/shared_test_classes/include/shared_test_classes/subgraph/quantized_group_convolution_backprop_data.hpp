// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

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
        ov::test::utils::QuantizationGranularity> quantGroupConvBackpropDataSpecificParams;
typedef std::tuple<
        quantGroupConvBackpropDataSpecificParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> quantGroupConvBackpropDataLayerTestParamsSet;

class QuantGroupConvBackpropDataLayerTest : public testing::WithParamInterface<quantGroupConvBackpropDataLayerTestParamsSet>,
                                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<quantGroupConvBackpropDataLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
