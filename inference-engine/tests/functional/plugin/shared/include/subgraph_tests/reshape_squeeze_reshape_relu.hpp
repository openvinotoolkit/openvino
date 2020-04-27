// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using ReshapeSqueezeReshapeReluTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes and squeeze_indices
        InferenceEngine::Precision,       //Network precision
        std::string,                      //Device name
        bool>;                            //Squeeze -> true, unsqueeze -> false

class ReshapeSqueezeReshapeRelu
        : public testing::WithParamInterface<ReshapeSqueezeReshapeReluTuple>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
