// Copyright (C) 2018-2023 Intel Corporation
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

#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

static std::map<ngraph::helpers::ActivationTypes, std::string> activationNames = {
        {ngraph::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,                  "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,                   "Exp"},
        {ngraph::helpers::ActivationTypes::Log,                   "Log"},
        {ngraph::helpers::ActivationTypes::Sign,                  "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,                   "Abs"},
        {ngraph::helpers::ActivationTypes::Clamp,                 "Clamp"},
        {ngraph::helpers::ActivationTypes::Negative,              "Negative"},
        {ngraph::helpers::ActivationTypes::Acos,                  "Acos"},
        {ngraph::helpers::ActivationTypes::Acosh,                 "Acosh"},
        {ngraph::helpers::ActivationTypes::Asin,                  "Asin"},
        {ngraph::helpers::ActivationTypes::Asinh,                 "Asinh"},
        {ngraph::helpers::ActivationTypes::Atan,                  "Atan"},
        {ngraph::helpers::ActivationTypes::Atanh,                  "Atanh"},
        {ngraph::helpers::ActivationTypes::Cos,                   "Cos"},
        {ngraph::helpers::ActivationTypes::Cosh,                  "Cosh"},
        {ngraph::helpers::ActivationTypes::Floor,                 "Floor"},
        {ngraph::helpers::ActivationTypes::Sin,                   "Sin"},
        {ngraph::helpers::ActivationTypes::Sinh,                  "Sinh"},
        {ngraph::helpers::ActivationTypes::Sqrt,                  "Sqrt"},
        {ngraph::helpers::ActivationTypes::Tan,                   "Tan"},
        {ngraph::helpers::ActivationTypes::Elu,                   "Elu"},
        {ngraph::helpers::ActivationTypes::Erf,                   "Erf"},
        {ngraph::helpers::ActivationTypes::HardSigmoid,           "HardSigmoid"},
        {ngraph::helpers::ActivationTypes::Selu,                  "Selu"},
        {ngraph::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,                  "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,                   "Exp"},
        {ngraph::helpers::ActivationTypes::Log,                   "Log"},
        {ngraph::helpers::ActivationTypes::Sign,                  "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,                   "Abs"},
        {ngraph::helpers::ActivationTypes::Gelu,                  "Gelu"},
        {ngraph::helpers::ActivationTypes::Ceiling,               "Ceiling"},
        {ngraph::helpers::ActivationTypes::PReLu,                 "PReLu"},
        {ngraph::helpers::ActivationTypes::Mish,                  "Mish"},
        {ngraph::helpers::ActivationTypes::HSwish,                "HSwish"},
        {ngraph::helpers::ActivationTypes::SoftPlus,              "SoftPlus"},
        {ngraph::helpers::ActivationTypes::Swish,                 "Swish"},
        {ngraph::helpers::ActivationTypes::HSigmoid,              "HSigmoid"},
        {ngraph::helpers::ActivationTypes::RoundHalfToEven,       "RoundHalfToEven"},
        {ngraph::helpers::ActivationTypes::RoundHalfAwayFromZero, "RoundHalfAwayFromZero"},
        {ngraph::helpers::ActivationTypes::GeluErf,               "GeluErf"},
        {ngraph::helpers::ActivationTypes::GeluTanh,              "GeluTanh"},
        {ngraph::helpers::ActivationTypes::SoftSign,              "SoftSign"},
};

typedef std::tuple<
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>, // Activation type and constant value
        InferenceEngine::Precision,
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
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
protected:
    void SetUp() override;

private:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void generateActivationBlob(std::vector<float> constantsValue);
    ngraph::ParameterVector createActivationParams(
        ngraph::element::Type ngPrc, std::vector<size_t> inShape = {});

private:
    std::vector<float> constantsValue;
};

class ActivationDynamicLayerTest : public ActivationLayerTest {
public:
    std::unordered_set<size_t> static_dims;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
