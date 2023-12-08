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

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        std::vector<ptrdiff_t>,
        std::vector<ptrdiff_t>,
        InferenceEngine::SizeVector,
        size_t,
        size_t,
        ngraph::op::PadType> groupConvSpecificParams;
typedef std::tuple<
        groupConvSpecificParams,
        InferenceEngine::Precision,
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> groupConvLayerTestParamsSet;

class GroupConvolutionLayerTest : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                                  virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
