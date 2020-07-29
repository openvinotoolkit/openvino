// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,        //Network precision
        std::string,                       //Device name
        std::map<std::string, std::string> //Configuration
> ConcatSplitReluTuple;

class ConcatSplitRelu
        : public testing::WithParamInterface<ConcatSplitReluTuple>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSplitReluTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
