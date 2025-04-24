// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/onnx/extension/op.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::frontend;

using ONNXOpExtensionTest = FrontEndOpExtensionTest;

static const std::string translator_name = "Relu";

class Relu1 : public Relu {
public:
    OPENVINO_OP("CustomRelu_1");
    OPENVINO_FRAMEWORK_MAP(onnx)
};

class Relu2 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(onnx, "CustomRelu_2")
};

class Relu3 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(onnx,
                           "CustomRelu_3",
                           {{"ov_attribute_1", "fw_attribute_1"}, {"ov_attribute_2", "fw_attribute_2"}})
};

class Relu4 : public Relu {
public:
    OPENVINO_FRAMEWORK_MAP(onnx,
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
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    // use core OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<Relu1>>(),
                                                                   std::make_shared<ov::OpExtension<Relu2>>(),
                                                                   std::make_shared<ov::OpExtension<Relu3>>(),
                                                                   std::make_shared<ov::OpExtension<Relu4>>()};
    return res;
}

static OpExtensionFEParam getTestDataOpExtensionViaONNXConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    // use ov::frontend::onnx OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        std::make_shared<ov::frontend::onnx::OpExtension<>>("CustomRelu_5"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("ov_CustomRelu_6", "fw_CustomRelu_6"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
            "ov_CustomRelu_7",
            "fw_CustomRelu_7",
            std::map<std::string, std::string>{{"ov_attribute_1", "fw_attribute_1"},
                                               {"ov_attribute_2", "fw_attribute_2"}}),
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
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
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
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

INSTANTIATE_TEST_SUITE_P(ONNXOpExtensionTestViaUserClass,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaUserClass()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ONNXOpExtensionViaONNXConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaONNXConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ONNXOpExtensionViaCommonConstructor,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestDataOpExtensionViaCommonConstructor()),
                         FrontEndOpExtensionTest::getTestCaseName);

TEST(ONNXOpExtensionViaCommonConstructor, onnx_op_extension_via_template_arg_with_custom_domain) {
    const auto ext =
        std::make_shared<ov::frontend::onnx::OpExtension<ov::op::v0::Relu>>("CustomRelu", "my_custom_domain");

    auto fe = std::make_shared<ov::frontend::onnx::FrontEnd>();
    fe->add_extension(ext);

    const auto input_model = fe->load(ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({TEST_ONNX_MODELS_DIRNAME, "relu_custom_domain.onnx"}).string()));

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(fe->convert(input_model));
}

TEST(ONNXOpExtensionViaCommonConstructor, onnx_op_extension_via_ov_type_name_with_custom_domain) {
    const auto ext =
        std::make_shared<ov::frontend::onnx::OpExtension<>>("opset1::Relu", "CustomRelu", "my_custom_domain");

    auto fe = std::make_shared<ov::frontend::onnx::FrontEnd>();
    fe->add_extension(ext);

    const auto input_model = fe->load(ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({TEST_ONNX_MODELS_DIRNAME, "relu_custom_domain.onnx"}).string()));

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(fe->convert(input_model));
}
