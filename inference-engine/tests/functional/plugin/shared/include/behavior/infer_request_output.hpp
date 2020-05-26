// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "multi-device/multi_device_config.hpp"

namespace LayerTestsDefinitions {
    typedef std::tuple<
            InferenceEngine::Precision,         // Network precision
            std::string,                        // Device name
            std::map<std::string, std::string>  // Config
    > InferRequestOutputParams;

class InferRequestOutputTests : public testing::WithParamInterface<InferRequestOutputParams>,
        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestOutputParams> obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace LayerTestsDefinitions