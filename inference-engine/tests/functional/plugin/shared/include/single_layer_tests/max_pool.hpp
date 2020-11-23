// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <ngraph/op/util/attr_types.hpp>

namespace LayerTestsDefinitions {

struct MaxPoolSpecificParams {
    InferenceEngine::SizeVector inputShape;
    std::vector<size_t> strides;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    std::vector<size_t> kernel;
    ngraph::op::RoundingType rounding_type;
    ngraph::op::PadType pad_type;
};

using MaxPoolLayerTestParams = std::tuple<
        MaxPoolSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
>;

class MaxPoolLayerTest : public testing::WithParamInterface<MaxPoolLayerTestParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolLayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
