// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using ActivationConcatsEltwiseParamsTuple = typename std::tuple<
    size_t,                             // input size
    size_t,                             // concat const size
    InferenceEngine::Precision,         // precision
    std::string,                        // device name
    std::map<std::string, std::string>  // configuration
>;


class ActivationConcatsEltwise : public testing::WithParamInterface<ActivationConcatsEltwiseParamsTuple>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ParamType> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
