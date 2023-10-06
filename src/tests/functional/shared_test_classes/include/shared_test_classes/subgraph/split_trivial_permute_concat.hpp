// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,        //Network precision
    std::string,                       //Device name
    std::vector<size_t>,               //Input sizes
    size_t,                            //Split axis
    size_t,                            //Concat axis
    std::map<std::string, std::string> //Configuration
> SplitTrivialPermuteConcatTuple;

class SplitTrivialPermuteConcatTest
    : public testing::WithParamInterface<SplitTrivialPermuteConcatTuple>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitTrivialPermuteConcatTuple>& obj);
protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
