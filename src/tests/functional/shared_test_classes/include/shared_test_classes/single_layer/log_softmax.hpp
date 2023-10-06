// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using logSoftmaxLayerTestParams = std::tuple<
        InferenceEngine::Precision,         // netPrecision
        InferenceEngine::Precision,         // Input precision
        InferenceEngine::Precision,         // Output precision
        InferenceEngine::Layout,            // Input layout
        InferenceEngine::Layout,            // Output layout
        InferenceEngine::SizeVector,        // inputShape
        int64_t,                            // axis
        std::string,                        // targetDevice
        std::map<std::string, std::string>  // config
>;

class LogSoftmaxLayerTest : public testing::WithParamInterface<logSoftmaxLayerTestParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<logSoftmaxLayerTestParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
