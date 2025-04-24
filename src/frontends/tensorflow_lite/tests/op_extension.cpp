// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/tensorflow_lite/extension/op.hpp"
#include "tf_utils.hpp"

using namespace ov::frontend;

using TFLiteOpExtensionTest = FrontEndOpExtensionTest;

class Relu1 : public Relu {
public:
    OPENVINO_OP("CustomRelu_1");
    OPENVINO_FRAMEWORK_MAP(tensorflow_lite)
};

class Relu2 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(tensorflow_lite, "CustomRelu_2")
};

class Relu3 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(tensorflow_lite,
                           "CustomRelu_3",
                           {{"ov_attribute_1", "fw_attribute_1"}, {"ov_attribute_2", "fw_attribute_2"}})
};

class Relu4 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(tensorflow_lite,
                           "CustomRelu_4",
                           {{"ov_attribute_1", "fw_attribute_1"}, {"ov_attribute_2", "fw_attribute_2"}},
                           {
                               {"ov_attribute_str", "string"},
                               {"ov_attribute_int", 4},
                               {"ov_attribute_bool", true},
                               {"ov_attribute_float", 4.f},
                               {"ov_attribute_vec_string", std::vector<std::string>{"str1", "str2", "str3"}},
                               {"ov_attribute_vec_int", std::vector<int>{1, 2, 3, 4, 5, 6, 7}},
                               {"ov_attribute_vec_bool", std::vector<bool>{true, false, true}},
                               {"ov_attribute_vec_float", std::vector<float>{1., 2., 3., 4., 5., 6., 7.}},
                           })
};

static OpExtensionFEParam getTestDataOpExtensionViaUserClass() {
    OpExtensionFEParam res;
    res.m_frontEndName = TF_LITE_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.tflite";
    // use core OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<Relu1>>(),
                                                                   std::make_shared<ov::OpExtension<Relu2>>(),
                                                                   std::make_shared<ov::OpExtension<Relu3>>(),
                                                                   std::make_shared<ov::OpExtension<Relu4>>()};
    return res;
}

static OpExtensionFEParam getTestDataOpExtensionViaTFConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = TF_LITE_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    // use ov::frontend::tensorflow OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        std::make_shared<ov::frontend::tensorflow_lite::OpExtension<>>("CustomRelu_5"),
        std::make_shared<ov::frontend::tensorflow_lite::OpExtension<>>("ov_CustomRelu_6", "fw_CustomRelu_6"),
        std::make_shared<ov::frontend::tensorflow_lite::OpExtension<>>(
            "ov_CustomRelu_7",
            "fw_CustomRelu_7",
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}}),
        std::make_shared<ov::frontend::tensorflow_lite::OpExtension<>>(
            "ov_CustomRelu_8",
            "fw_CustomRelu_8",
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}},
            std::map<std::string, ov::Any>{
                {"ov_attribute_str", "string"},
                {"ov_attribute_int", 4},
                {"ov_attribute_bool", true},
                {"ov_attribute_float", 4.f},
                {"ov_attribute_vec_string", std::vector<std::string>{"str1", "str2", "str3"}},
                {"ov_attribute_vec_int", std::vector<int>{1, 2, 3, 4, 5, 6, 7}},
                {"ov_attribute_vec_bool", std::vector<bool>{true, false, true}},
                {"ov_attribute_vec_float", std::vector<float>{1., 2., 3., 4., 5., 6., 7.}},
            })};
    return res;
}

static OpExtensionFEParam getTestDataOpExtensionViaCommonConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = TF_LITE_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.tflite";
    // use ov::frontend::OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        std::make_shared<ov::frontend::OpExtension<>>("CustomRelu_9"),
        std::make_shared<ov::frontend::OpExtension<>>("ov_CustomRelu_10", "fw_CustomRelu_10"),
        std::make_shared<ov::frontend::OpExtension<>>(
            "ov_CustomRelu_11",
            "fw_CustomRelu_11",
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}}),
        std::make_shared<ov::frontend::OpExtension<>>(
            "ov_CustomRelu_12",
            "fw_CustomRelu_12",
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}},
            std::map<std::string, ov::Any>{
                {"ov_attribute_str", "string"},
                {"ov_attribute_int", 4},
                {"ov_attribute_bool", true},
                {"ov_attribute_float", 4.f},
                {"ov_attribute_vec_string", std::vector<std::string>{"str1", "str2", "str3"}},
                {"ov_attribute_vec_int", std::vector<int>{1, 2, 3, 4, 5, 6, 7}},
                {"ov_attribute_vec_bool", std::vector<bool>{true, false, true}},
                {"ov_attribute_vec_float", std::vector<float>{1., 2., 3., 4., 5., 6., 7.}},
            })};
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFLiteOpExtensionTestViaUserClass,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaUserClass()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TFOpExtensionViaTFConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaTFConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TFOpExtensionViaCommonConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaCommonConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);
