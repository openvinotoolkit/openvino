// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <set>


#include "shared_test_classes/base/ov_subgraph.hpp"

#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using ov::test::utils::ActivationTypes;

static std::map<ActivationTypes, std::string> activationNames = {
        {ActivationTypes::Sigmoid,               "Sigmoid"},
        {ActivationTypes::Tanh,                  "Tanh"},
        {ActivationTypes::Relu,                  "Relu"},
        {ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ActivationTypes::Exp,                   "Exp"},
        {ActivationTypes::Log,                   "Log"},
        {ActivationTypes::Sign,                  "Sign"},
        {ActivationTypes::Abs,                   "Abs"},
        {ActivationTypes::Clamp,                 "Clamp"},
        {ActivationTypes::Negative,              "Negative"},
        {ActivationTypes::Acos,                  "Acos"},
        {ActivationTypes::Acosh,                 "Acosh"},
        {ActivationTypes::Asin,                  "Asin"},
        {ActivationTypes::Asinh,                 "Asinh"},
        {ActivationTypes::Atan,                  "Atan"},
        {ActivationTypes::Atanh,                  "Atanh"},
        {ActivationTypes::Cos,                   "Cos"},
        {ActivationTypes::Cosh,                  "Cosh"},
        {ActivationTypes::Floor,                 "Floor"},
        {ActivationTypes::Sin,                   "Sin"},
        {ActivationTypes::Sinh,                  "Sinh"},
        {ActivationTypes::Sqrt,                  "Sqrt"},
        {ActivationTypes::Tan,                   "Tan"},
        {ActivationTypes::Elu,                   "Elu"},
        {ActivationTypes::Erf,                   "Erf"},
        {ActivationTypes::HardSigmoid,           "HardSigmoid"},
        {ActivationTypes::Selu,                  "Selu"},
        {ActivationTypes::Sigmoid,               "Sigmoid"},
        {ActivationTypes::Tanh,                  "Tanh"},
        {ActivationTypes::Relu,                  "Relu"},
        {ActivationTypes::Exp,                   "Exp"},
        {ActivationTypes::Log,                   "Log"},
        {ActivationTypes::Sign,                  "Sign"},
        {ActivationTypes::Abs,                   "Abs"},
        {ActivationTypes::Gelu,                  "Gelu"},
        {ActivationTypes::Ceiling,               "Ceiling"},
        {ActivationTypes::PReLu,                 "PReLu"},
        {ActivationTypes::Mish,                  "Mish"},
        {ActivationTypes::HSwish,                "HSwish"},
        {ActivationTypes::SoftPlus,              "SoftPlus"},
        {ActivationTypes::Swish,                 "Swish"},
        {ActivationTypes::HSigmoid,              "HSigmoid"},
        {ActivationTypes::RoundHalfToEven,       "RoundHalfToEven"},
        {ActivationTypes::RoundHalfAwayFromZero, "RoundHalfAwayFromZero"},
        {ActivationTypes::GeluErf,               "GeluErf"},
        {ActivationTypes::GeluTanh,              "GeluTanh"},
        {ActivationTypes::SoftSign,              "SoftSign"},
        {ActivationTypes::IsInf,                 "IsInf"},
        {ActivationTypes::IsFinite,              "IsFinite"},
        {ActivationTypes::IsNaN,                 "IsNaN"},
        {ActivationTypes::LogicalNot,            "LogicalNot"},
};

typedef std::tuple<
        std::pair<ActivationTypes, std::vector<float>>,  // Activation type and constant value
        ov::element::Type,                               // Model type
        std::pair<std::vector<InputShape>,               // Input shapes
        ov::Shape>,                                      // 2nd input const shape
        std::string> activationParams;

class ActivationLayerTest : public testing::WithParamInterface<activationParams>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);

protected:
    //TO DO, to be removed after 125993
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;
};

class ActivationParamLayerTest : public ActivationLayerTest {
protected:
    //TO DO, to be removed after 125993
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
