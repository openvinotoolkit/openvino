// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <gtest/gtest.h>


#include "ie_core.hpp"
#include "ie_precision.hpp"
#include "details/ie_exception.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/runtime/reference/relu.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {

static std::map<ngraph::helpers::ActivationTypes, std::string> activationNames = {
        {ngraph::helpers::ActivationTypes::Sigmoid,   "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,      "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,      "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu, "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,       "Exp"},
        {ngraph::helpers::ActivationTypes::Log,       "Log"},
        {ngraph::helpers::ActivationTypes::Sign,      "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,       "Abs"}
};

typedef std::tuple<
        ngraph::helpers::ActivationTypes,
        InferenceEngine::Precision,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        std::string> activationParams;

class ActivationLayerTest
        : public LayerTestsUtils::LayerTestsCommonClass<activationParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    ngraph::helpers::ActivationTypes activationType;

    void SetUp();
};

}  // namespace LayerTestsDefinitions
