// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace LayerTestsDefinitions {

using GreaterTestParam = typename std::tuple<
        std::vector<InferenceEngine::SizeVector>,  // Input shapes
        InferenceEngine::Precision,                // Input precision
        InferenceEngine::Precision,                // Output precision
        LayerTestsUtils::TargetDevice>;            // Config

class GreaterLayerTest : public testing::WithParamInterface<GreaterTestParam>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GreaterTestParam>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
