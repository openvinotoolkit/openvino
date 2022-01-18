// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using EltwiseReshapeActivationParams = typename std::tuple<
    std::vector<std::vector<size_t>>,   // input shape and shape after reshape
    InferenceEngine::Precision,         // precision
    std::string,                        // device name
    std::map<std::string, std::string>  // configuration
>;

class EltwiseReshapeActivation : public testing::WithParamInterface<EltwiseReshapeActivationParams>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
