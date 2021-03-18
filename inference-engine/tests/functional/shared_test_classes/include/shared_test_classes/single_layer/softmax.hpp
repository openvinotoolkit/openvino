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

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using softMaxLayerTestParams = std::tuple<
        InferenceEngine::Precision,         // netPrecision
        InferenceEngine::Precision,         // Input precision
        InferenceEngine::Precision,         // Output precision
        InferenceEngine::Layout,            // Input layout
        InferenceEngine::Layout,            // Output layout
        InferenceEngine::SizeVector,        // inputShape
        size_t,                             // axis
        std::string,                        // targetDevice
        std::map<std::string, std::string>  // config
>;

class SoftMaxLayerTest : public testing::WithParamInterface<softMaxLayerTestParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<softMaxLayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
