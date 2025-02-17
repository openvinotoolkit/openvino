// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/paddle/extension/op.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/runtime/core.hpp"
#include "paddle_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend;

class Relu1 : public Relu {
public:
    OPENVINO_OP("relu");
    OPENVINO_FRAMEWORK_MAP(paddle, {"X"}, {"Out"})
};

class Relu2 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(paddle, {"X"}, {"Out"}, "CustomRelu_2")
};

class Relu3 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(paddle,
                           {"X"},
                           {"Out"},
                           "CustomRelu_3",
                           {{"ov_attribute_1", "fw_attribute_1"}, {"ov_attribute_2", "fw_attribute_2"}})
};

class Relu4 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(paddle,
                           {"X"},
                           {"Out"},
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
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    // use core OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<Relu1>>(),
                                                                   std::make_shared<ov::OpExtension<Relu2>>(),
                                                                   std::make_shared<ov::OpExtension<Relu3>>(),
                                                                   std::make_shared<ov::OpExtension<Relu4>>()};
    return res;
}

static OpExtensionFEParam getTestDataOpExtensionViaPaddleConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    // use ov::frontend::paddle OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        std::make_shared<ov::frontend::paddle::OpExtension<>>("CustomRelu_5",
                                                              std::vector<std::string>{"X"},
                                                              std::vector<std::string>{"Out"}),
        std::make_shared<ov::frontend::paddle::OpExtension<>>("ov_CustomRelu_6",
                                                              "fw_CustomRelu_6",
                                                              std::vector<std::string>{"X"},
                                                              std::vector<std::string>{"Out"}),
        std::make_shared<ov::frontend::paddle::OpExtension<>>(
            "ov_CustomRelu_7",
            "fw_CustomRelu_7",
            std::vector<std::string>{"X"},
            std::vector<std::string>{"Out"},
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}}),
        std::make_shared<ov::frontend::paddle::OpExtension<>>(
            "ov_CustomRelu_8",
            "fw_CustomRelu_8",
            std::vector<std::string>{"X"},
            std::vector<std::string>{"Out"},
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
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    // use ov::frontend::OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        std::make_shared<ov::frontend::OpExtension<>>("CustomRelu_9",
                                                      std::vector<std::string>{"X"},
                                                      std::vector<std::string>{"Out"}),
        std::make_shared<ov::frontend::OpExtension<>>("ov_CustomRelu_10",
                                                      "fw_CustomRelu_10",
                                                      std::vector<std::string>{"X"},
                                                      std::vector<std::string>{"Out"}),
        std::make_shared<ov::frontend::OpExtension<>>(
            "ov_CustomRelu_11",
            "fw_CustomRelu_11",
            std::vector<std::string>{"X"},
            std::vector<std::string>{"Out"},
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}}),
        std::make_shared<ov::frontend::OpExtension<>>(
            "ov_CustomRelu_12",
            "fw_CustomRelu_12",
            std::vector<std::string>{"X"},
            std::vector<std::string>{"Out"},
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

INSTANTIATE_TEST_SUITE_P(PaddleOpExtensionTestViaUserClass,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaUserClass()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(PaddleOpExtensionViaPaddleConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaPaddleConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(PaddleOpExtensionViaCommonConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaCommonConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);

TEST(FrontEndOpExtensionTest, paddle_opextension_relu) {
    ov::Core core;
    const auto extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<Relu1>>()};
    core.add_extension(extensions);
    std::string m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    std::string m_modelName = "relu/relu.pdmodel";
    auto model = core.read_model(FrontEndTestUtils::make_model_path(m_modelsPath + m_modelName));
    bool has_relu = false;
    for (const auto& op : model->get_ops()) {
        std::string name = op->get_type_info().name;
        std::string version = op->get_type_info().version_id;
        if (name == "relu" && version == "extension")
            has_relu = true;
    }
    EXPECT_TRUE(has_relu);
}
