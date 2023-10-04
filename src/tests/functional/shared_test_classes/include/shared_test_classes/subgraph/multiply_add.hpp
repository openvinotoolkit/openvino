// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

using MultiplyAddParamsTuple = typename std::tuple<
        std::vector<size_t>,              //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string>;                     //Device name

class MultiplyAddLayerTest:
        public testing::WithParamInterface<MultiplyAddParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyAddParamsTuple> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
