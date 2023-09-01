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

static std::map<ov::helpers::ActivationTypes, std::string> activationNames = {
        {ov::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ov::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ov::helpers::ActivationTypes::Relu,                  "Relu"},
        {ov::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ov::helpers::ActivationTypes::Exp,                   "Exp"},
        {ov::helpers::ActivationTypes::Log,                   "Log"},
        {ov::helpers::ActivationTypes::Sign,                  "Sign"},
        {ov::helpers::ActivationTypes::Abs,                   "Abs"},
        {ov::helpers::ActivationTypes::Clamp,                 "Clamp"},
        {ov::helpers::ActivationTypes::Negative,              "Negative"},
        {ov::helpers::ActivationTypes::Acos,                  "Acos"},
        {ov::helpers::ActivationTypes::Acosh,                 "Acosh"},
        {ov::helpers::ActivationTypes::Asin,                  "Asin"},
        {ov::helpers::ActivationTypes::Asinh,                 "Asinh"},
        {ov::helpers::ActivationTypes::Atan,                  "Atan"},
        {ov::helpers::ActivationTypes::Atanh,                  "Atanh"},
        {ov::helpers::ActivationTypes::Cos,                   "Cos"},
        {ov::helpers::ActivationTypes::Cosh,                  "Cosh"},
        {ov::helpers::ActivationTypes::Floor,                 "Floor"},
        {ov::helpers::ActivationTypes::Sin,                   "Sin"},
        {ov::helpers::ActivationTypes::Sinh,                  "Sinh"},
        {ov::helpers::ActivationTypes::Sqrt,                  "Sqrt"},
        {ov::helpers::ActivationTypes::Tan,                   "Tan"},
        {ov::helpers::ActivationTypes::Elu,                   "Elu"},
        {ov::helpers::ActivationTypes::Erf,                   "Erf"},
        {ov::helpers::ActivationTypes::HardSigmoid,           "HardSigmoid"},
        {ov::helpers::ActivationTypes::Selu,                  "Selu"},
        {ov::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ov::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ov::helpers::ActivationTypes::Relu,                  "Relu"},
        {ov::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ov::helpers::ActivationTypes::Exp,                   "Exp"},
        {ov::helpers::ActivationTypes::Log,                   "Log"},
        {ov::helpers::ActivationTypes::Sign,                  "Sign"},
        {ov::helpers::ActivationTypes::Abs,                   "Abs"},
        {ov::helpers::ActivationTypes::Gelu,                  "Gelu"},
        {ov::helpers::ActivationTypes::Ceiling,               "Ceiling"},
        {ov::helpers::ActivationTypes::PReLu,                 "PReLu"},
        {ov::helpers::ActivationTypes::Mish,                  "Mish"},
        {ov::helpers::ActivationTypes::HSwish,                "HSwish"},
        {ov::helpers::ActivationTypes::SoftPlus,              "SoftPlus"},
        {ov::helpers::ActivationTypes::Swish,                 "Swish"},
        {ov::helpers::ActivationTypes::HSigmoid,              "HSigmoid"},
        {ov::helpers::ActivationTypes::RoundHalfToEven,       "RoundHalfToEven"},
        {ov::helpers::ActivationTypes::RoundHalfAwayFromZero, "RoundHalfAwayFromZero"},
        {ov::helpers::ActivationTypes::GeluErf,               "GeluErf"},
        {ov::helpers::ActivationTypes::GeluTanh,              "GeluTanh"},
        {ov::helpers::ActivationTypes::SoftSign,              "SoftSign"},
};

typedef std::tuple<
        std::pair<ov::helpers::ActivationTypes, std::vector<float>>, // Activation type and constant value
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
    ov::helpers::ActivationTypes activationType;
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
