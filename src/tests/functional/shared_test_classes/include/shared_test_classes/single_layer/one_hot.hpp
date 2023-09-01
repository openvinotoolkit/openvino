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
namespace LayerTestsDefinitions {
typedef std::tuple<
        ngraph::element::Type,          // depth type (any integer type)
        int64_t,                        // depth value
        ngraph::element::Type,          // On & Off values type (any supported type)
        float,                          // OnValue
        float,                          // OffValue
        int64_t,                        // axis
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Target device name
> oneHotLayerTestParamsSet;

class OneHotLayerTest : public testing::WithParamInterface<oneHotLayerTestParamsSet>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHotLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
