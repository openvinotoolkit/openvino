// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
namespace LayerTestsDefinitions {
typedef std::tuple<
        std::pair<ngraph::element::Type, int64_t>,   // <any_int_type, depth>
        std::pair<ngraph::element::Type, float>,    // <any_type, on_value>
        std::pair<ngraph::element::Type, float>,    // <any_type, off_value>
        int64_t,                       // axis
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> oneHotLayerTestParamsSet;

class OneHotLayerTest : public testing::WithParamInterface<oneHotLayerTestParamsSet>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<oneHotLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
