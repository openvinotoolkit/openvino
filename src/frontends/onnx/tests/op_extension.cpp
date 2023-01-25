// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/enum_names.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/onnx/extension/op.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/op/relu.hpp"
#include "so_extension.hpp"

using namespace ov::frontend;

class TestOperation1 : public TestOperation {
public:
    OPENVINO_OP("TestOperation1");
    OPENVINO_FRAMEWORK_MAP(onnx)
};

class TestOperation2 : public TestOperation {
public:
    OPENVINO_FRAMEWORK_MAP(onnx, "TestOperation2")
};

// 1:1 mapping without type conversion
// double/int64 attributes from onnx proto are converted to double/int64 ov attributes
// double -> double, int64 -> int64, vector<double> -> vector<double>, vector<int64> -> vector<int64>
// string -> string
class TestOperation3 : public TestOperation {
public:
    OPENVINO_FRAMEWORK_MAP(onnx,
                           "TestOperation3",
                           // the values of these attributes will be read
                           // and set to the original values from framework model
                           {
                               {"ov_attribute_double", "fw_attribute_float"},
                               {"ov_attribute_str", "fw_attribute_str"},
                               {"ov_attribute_int64", "fw_attribute_int"},
                               {"ov_attribute_vec_str", "fw_attribute_vec_str"},
                               {"ov_attribute_vec_int64", "fw_attribute_vec_int"},
                               {"ov_attribute_vec_double", "fw_attribute_vec_float"},
                           })
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("ov_attribute_double", ov_attribute_double);
        visitor.on_attribute("ov_attribute_str", ov_attribute_str);
        visitor.on_attribute("ov_attribute_int64", ov_attribute_int64);
        visitor.on_attribute("ov_attribute_vec_str", ov_attribute_vec_str);
        visitor.on_attribute("ov_attribute_vec_int64", ov_attribute_vec_int64);
        visitor.on_attribute("ov_attribute_vec_double", ov_attribute_vec_double);
        return true;
    }

private:
    double ov_attribute_double = 0;
    std::string ov_attribute_str;
    int64_t ov_attribute_int64 = 0;
    std::vector<std::string> ov_attribute_vec_str;
    std::vector<int64_t> ov_attribute_vec_int64;
    std::vector<double> ov_attribute_vec_double;
};

enum class TestEnum { Value1, Value2, Value3 };

template <>
NGRAPH_API ov::EnumNames<TestEnum>& ov::EnumNames<TestEnum>::get() {
    static auto enum_names =
        EnumNames<TestEnum>("TestEnum",
                            {{"Value1", TestEnum::Value1}, {"Value2", TestEnum::Value2}, {"Value3", TestEnum::Value3}});
    return enum_names;
}

template <>
class OPENVINO_API ov::AttributeAdapter<TestEnum> : public EnumAttributeAdapterBase<TestEnum> {
public:
    explicit AttributeAdapter(TestEnum& value) : EnumAttributeAdapterBase<TestEnum>(value) {}

    OPENVINO_RTTI("AttributeAdapter<TestEnum>");
};

std::ostream& operator<<(std::ostream& s, const ngraph::op::PadMode& type) {
    return s << ov::as_string(type);
}

// 1:1 mapping with type conversion
// double/int64 attributes from onnx proto are converted to float/int32/bool ov attributes
// double -> float, int64 -> int32, vector<double> -> vector<float>, vector<int64> -> vector<int32>
// int64 -> bool, string -> enum
class TestOperation4 : public TestOperation {
public:
    OPENVINO_FRAMEWORK_MAP(onnx,
                           "TestOperation4",
                           // the values of these attributes will be read
                           // and set to the original values from framework model
                           {
                               {"ov_attribute_float", "fw_attribute_float"},
                               {"ov_attribute_str", "fw_attribute_str"},
                               {"ov_attribute_int32", "fw_attribute_int"},
                               {"ov_attribute_bool", "fw_attribute_bool"},
                               {"ov_attribute_vec_int", "fw_attribute_vec_int"},
                               {"ov_attribute_vec_float", "fw_attribute_vec_float"},
                               {"ov_attribute_enum", "fw_attribute_enum"},
                           })
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("ov_attribute_float", ov_attribute_float);
        visitor.on_attribute("ov_attribute_str", ov_attribute_str);
        visitor.on_attribute("ov_attribute_int32", ov_attribute_int32);
        visitor.on_attribute("ov_attribute_bool", ov_attribute_bool);
        visitor.on_attribute("ov_attribute_vec_int32", ov_attribute_vec_int32);
        visitor.on_attribute("ov_attribute_vec_float", ov_attribute_vec_float);
        visitor.on_attribute("ov_attribute_enum", ov_attribute_enum);
        return true;
    }

private:
    float ov_attribute_float = 0;
    std::string ov_attribute_str;
    int32_t ov_attribute_int32 = 0;
    bool ov_attribute_bool = false;
    std::vector<int32_t> ov_attribute_vec_int32;
    std::vector<float> ov_attribute_vec_float;
    TestEnum ov_attribute_enum = TestEnum::Value1;
};

// 1:1 mapping with the provided values for the attributes
class TestOperation5 : public TestOperation {
public:
    OPENVINO_FRAMEWORK_MAP(onnx,
                           "TestOperation5",
                           {},
                           {
                               // these attributes have to be set to these values
                               {"ov_attribute_str", "string"},
                               {"ov_attribute_int64", 4},
                               {"ov_attribute_int32", 4},
                               {"ov_attribute_bool", true},
                               {"ov_attribute_double", 4.f},
                               {"ov_attribute_float", 4.f},
                               {"ov_attribute_vec_string", std::vector<std::string>{"str1", "str2", "str3"}},
                               {"ov_attribute_vec_int64", std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7}},
                               {"ov_attribute_vec_int32", std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7}},
                               {"ov_attribute_vec_double", std::vector<double>{1., 2., 3., 4., 5., 6., 7.}},
                               {"ov_attribute_vec_float", std::vector<float>{1., 2., 3., 4., 5., 6., 7.}},
                           })
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("ov_attribute_str", ov_attribute_str);
        visitor.on_attribute("ov_attribute_int64", ov_attribute_int64);
        visitor.on_attribute("ov_attribute_int32", ov_attribute_int32);
        visitor.on_attribute("ov_attribute_bool", ov_attribute_bool);
        visitor.on_attribute("ov_attribute_double", ov_attribute_double);
        visitor.on_attribute("ov_attribute_float", ov_attribute_float);
        visitor.on_attribute("ov_attribute_vec_str", ov_attribute_vec_str);
        visitor.on_attribute("ov_attribute_vec_int64", ov_attribute_vec_int64);
        visitor.on_attribute("ov_attribute_vec_int32", ov_attribute_vec_int32);
        visitor.on_attribute("ov_attribute_vec_double", ov_attribute_vec_double);
        visitor.on_attribute("ov_attribute_vec_float", ov_attribute_vec_float);
        return true;
    }

private:
    std::string ov_attribute_str;
    int64_t ov_attribute_int64 = 0;
    int64_t ov_attribute_int32 = 0;
    bool ov_attribute_bool = false;
    double ov_attribute_double = 0;
    double ov_attribute_float = 0;
    std::vector<std::string> ov_attribute_vec_str;
    std::vector<int64_t> ov_attribute_vec_int64;
    std::vector<int32_t> ov_attribute_vec_int32;
    std::vector<double> ov_attribute_vec_double;
    std::vector<float> ov_attribute_vec_float;
};

// cast fw operations to new created TestOperationN
static OpExtensionFEParam getTestDataOpExtensionViaUserClass() {
    OpExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "extensions/fe_op_extension_model_1.onnx";
    // use core OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<TestOperation1>>(),
                                                                   std::make_shared<ov::OpExtension<TestOperation2>>(),
                                                                   std::make_shared<ov::OpExtension<TestOperation3>>(),
                                                                   std::make_shared<ov::OpExtension<TestOperation4>>(),
                                                                   std::make_shared<ov::OpExtension<TestOperation5>>()};
    return res;
}

// cast fw operations to existed ov operations
static OpExtensionFEParam getTestDataOpExtensionViaONNXConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "extensions/fe_op_extension_model_2.onnx";
    // use ov::frontend::onnx OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        // 1:1 mapping with same op and attribute names
        std::make_shared<ov::frontend::onnx::OpExtension<>>("Elu"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("Convert"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("IsInf"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("LogSoftmax"),
        // 1:1 mapping with different op and attribute names
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
            "Elu",
            "fw_Elu",
            std::map<std::string, std::string>({{"alpha", "fw_alpha"}})),
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
            "Convert",
            "fw_Convert",
            std::map<std::string, std::string>({{"destination_type", "fw_destination_type"}})),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("IsInf",
                                                            "fw_IsInf",
                                                            std::map<std::string, std::string>({
                                                                {"detect_negative", "fw_detect_negative"},
                                                                {"detect_positive", "fw_detect_positive"},
                                                            })),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("LogSoftmax",
                                                            "fw_LogSoftmax",
                                                            std::map<std::string, std::string>({{"axis", "fw_axis"}}))};

    return res;
}

// cast fw operations to existed ov operations
static OpExtensionFEParam getTestDataOpExtensionViaCommonConstructor() {
    OpExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "extensions/fe_op_extension_model_2.onnx";
    // use ov::frontend::OpExtension
    res.m_extensions = std::vector<std::shared_ptr<ov::Extension>>{
        // 1:1 mapping with same op and attribute names
        std::make_shared<ov::frontend::onnx::OpExtension<>>("Elu"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("Convert"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("IsInf"),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("LogSoftmax"),
        // 1:1 mapping with different op and attribute names
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
            "Elu",
            "fw_Elu",
            std::map<std::string, std::string>({{"alpha", "fw_alpha"}})),
        std::make_shared<ov::frontend::onnx::OpExtension<>>(
            "Convert",
            "fw_Elu",
            std::map<std::string, std::string>({{"destination_type", "fw_destination_type"}})),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("IsInf",
                                                            "fw_IsInf",
                                                            std::map<std::string, std::string>({
                                                                {"detect_negative", "fw_detect_negative"},
                                                                {"detect_positive", "fw_detect_positive"},
                                                            })),
        std::make_shared<ov::frontend::onnx::OpExtension<>>("LogSoftmax",
                                                            "fw_LogSoftmax",
                                                            std::map<std::string, std::string>({{"axis", "fw_axis"}}))};
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
    const auto ext = std::make_shared<onnx::OpExtension<ov::op::v0::Relu>>("CustomRelu", "my_custom_domain");

    auto fe = std::make_shared<ov::frontend::onnx::FrontEnd>();
    fe->add_extension(ext);

    const auto input_model = fe->load(CommonTestUtils::getModelFromTestModelZoo(
        ov::util::path_join({TEST_ONNX_MODELS_DIRNAME, "relu_custom_domain.onnx"})));

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(fe->convert(input_model));
}

TEST(ONNXOpExtensionViaCommonConstructor, onnx_op_extension_via_ov_type_name_with_custom_domain) {
    const auto ext = std::make_shared<onnx::OpExtension<>>("opset1::Relu", "CustomRelu", "my_custom_domain");

    auto fe = std::make_shared<ov::frontend::onnx::FrontEnd>();
    fe->add_extension(ext);

    const auto input_model = fe->load(CommonTestUtils::getModelFromTestModelZoo(
        ov::util::path_join({TEST_ONNX_MODELS_DIRNAME, "relu_custom_domain.onnx"})));

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(fe->convert(input_model));
}
