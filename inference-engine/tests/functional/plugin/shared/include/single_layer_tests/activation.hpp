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
        {ngraph::helpers::ActivationTypes::Sigmoid,     "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,        "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,        "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,   "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,         "Exp"},
        {ngraph::helpers::ActivationTypes::Log,         "Log"},
        {ngraph::helpers::ActivationTypes::Sign,        "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,         "Abs"},
        {ngraph::helpers::ActivationTypes::Gelu,        "Gelu"},
        {ngraph::helpers::ActivationTypes::Clamp,       "Clamp"},
        {ngraph::helpers::ActivationTypes::Negative,    "Negative"},
        {ngraph::helpers::ActivationTypes::Acos,        "Acos"},
        {ngraph::helpers::ActivationTypes::Asin,        "Asin"},
        {ngraph::helpers::ActivationTypes::Atan,        "Atan"},
        {ngraph::helpers::ActivationTypes::Cos,         "Cos"},
        {ngraph::helpers::ActivationTypes::Cosh,        "Cosh"},
        {ngraph::helpers::ActivationTypes::Floor,       "Floor"},
        {ngraph::helpers::ActivationTypes::Sin,         "Sin"},
        {ngraph::helpers::ActivationTypes::Sinh,        "Sinh"},
        {ngraph::helpers::ActivationTypes::Sqrt,        "Sqrt"},
        {ngraph::helpers::ActivationTypes::Tan,         "Tan"},
        {ngraph::helpers::ActivationTypes::Elu,         "Elu"},
        {ngraph::helpers::ActivationTypes::Erf,         "Erf"},
        {ngraph::helpers::ActivationTypes::HardSigmoid, "HardSigmoid"},
        {ngraph::helpers::ActivationTypes::Selu,        "Selu"},
        {ngraph::helpers::ActivationTypes::Sigmoid,     "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,        "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,        "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,   "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,         "Exp"},
        {ngraph::helpers::ActivationTypes::Log,         "Log"},
        {ngraph::helpers::ActivationTypes::Sign,        "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,         "Abs"},
        {ngraph::helpers::ActivationTypes::Gelu,        "Gelu"},
        {ngraph::helpers::ActivationTypes::Ceiling,     "Ceiling"},
        {ngraph::helpers::ActivationTypes::PReLu,       "PReLu"},
        {ngraph::helpers::ActivationTypes::Mish,        "Mish"},
};

typedef std::tuple<
        ngraph::helpers::ActivationTypes,
        InferenceEngine::Precision,
        std::pair<std::vector<size_t>, std::vector<size_t>>,
        std::string> activationParams;

class ActivationLayerTest : public testing::WithParamInterface<activationParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    ngraph::helpers::ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

class ActivationParamLayerTest : public ActivationLayerTest {
public:
    void Infer() override;

protected:
    void SetUp() override;

private:
    void generateActivationBlob();
    ngraph::ParameterVector createActivationParams(ngraph::element::Type ngPrc, std::vector<size_t> inShape = {});
};

}  // namespace LayerTestsDefinitions
