// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
    typedef std::tuple<
            InferenceEngine::Precision,         // Network precision
            std::vector<std::vector<size_t>>,   // Input shapes
            std::string,                        // Device name
            std::map<std::string, std::string>  // Config
            > addParams;

class AddLayerTest : public testing::WithParamInterface<addParams>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<addParams> obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
