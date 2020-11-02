// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using SplitConcatMemoryParamsTuple = typename std::tuple<
    std::vector<size_t>,         // input shapes
    InferenceEngine::Precision,  // precision
    int,                         // axis of split
    std::string                  // device name
>;


class SplitConcatMemory : public testing::WithParamInterface<SplitConcatMemoryParamsTuple>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ParamType> obj);

protected:
    void SetUp() override;

    int axis;
};

}  // namespace LayerTestsDefinitions