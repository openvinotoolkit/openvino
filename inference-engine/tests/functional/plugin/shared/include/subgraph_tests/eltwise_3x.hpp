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

typedef std::tuple<
        std::vector<std::vector<size_t>>,  // input shapes
        std::vector<InferenceEngine::Precision>,        //input precisions
        bool,         // input precisions
        std::string                       //Device name
> Eltwise3xTuple;

class Eltwise3x
        : public testing::WithParamInterface<Eltwise3xTuple>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Eltwise3xTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
