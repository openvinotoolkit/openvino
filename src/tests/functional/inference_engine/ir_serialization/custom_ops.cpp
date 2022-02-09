// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <file_utils.h>
#include <ie_api.h>
#include <ie_iextension.h>
#include <ie_network_reader.hpp>
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/serialize.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#error "IR_SERIALIZATION_MODELS_PATH is not defined"
#endif

#ifndef IE_BUILD_POSTFIX  // should be already defined by cmake
#error "IE_BUILD_POSTFIX is not defined"
#endif

static std::string get_extension_path() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("template_extension") + IE_BUILD_POSTFIX);
}

static std::string get_ov_extension_path() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("openvino_template_extension") + IE_BUILD_POSTFIX);
}

class CustomOpsSerializationTest : public ::testing::Test {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(CustomOpsSerializationTest, CustomOpUser_MO) {
    const std::string model = CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_op.xml");

    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InferenceEngine::Extension>(get_extension_path()));

    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

#ifdef ENABLE_OV_ONNX_FRONTEND

// This test will not work because template_extension for ONNX registers
// extension via `register_operator` function which registers operator
// is template_extension's copy of onnx_importer. So, extensions as
// a shared library for ONNX don't make sence in static OpenVINO build
#ifndef OPENVINO_STATIC_LIBRARY

TEST_F(CustomOpsSerializationTest, CustomOpUser_ONNXImporter) {
    const std::string model = CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_op.onnx");

    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InferenceEngine::Extension>(get_extension_path()));

    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

#endif  // OPENVINO_STATIC_LIBRARY

#endif  // ENABLE_OV_ONNX_FRONTEND

TEST_F(CustomOpsSerializationTest, CustomOpTransformation) {
    const std::string model = CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_op.xml");

    InferenceEngine::Core ie;
    auto extension = std::make_shared<InferenceEngine::Extension>(get_extension_path());
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
        m_out_xml_path, m_out_bin_path, extension->getOpSets(), ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

class FrameworkNodeExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static InferenceEngine::Version ExtensionDescription = {{1, 0}, "1.0", "framework_node_ext"};

        versionInfo = &ExtensionDescription;
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<ov::op::util::FrameworkNode>();
            opsets["util"] = opset;
        }
        return opsets;
    }

    void Unload() noexcept override {}
};

TEST_F(CustomOpsSerializationTest, CustomOpNoExtensions) {
    const std::string model = CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_op.xml");

    InferenceEngine::Core ie;
    auto extension = std::make_shared<FrameworkNodeExtension>();
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
        m_out_xml_path, m_out_bin_path, extension->getOpSets(), ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}

TEST_F(CustomOpsSerializationTest, CustomOpOVExtensions) {
    const std::string model =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_identity.xml");

    ov::Core core;
    core.add_extension(get_ov_extension_path());
    auto expected = core.read_model(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
        m_out_xml_path, m_out_bin_path, ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(expected);
    auto result = core.read_model(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result, expected, true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}

TEST_F(CustomOpsSerializationTest, CloneFrameworkNode) {
    const std::string model = CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "custom_op.xml");
    InferenceEngine::Core ie;
    auto extension = std::make_shared<FrameworkNodeExtension>();
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    auto clone = ov::clone_model(*expected.getFunction());

    const FunctionsComparator func_comparator = FunctionsComparator::with_default()
            .enable(FunctionsComparator::ATTRIBUTES)
            .enable(FunctionsComparator::CONST_VALUES)
            .enable(FunctionsComparator::PRECISIONS);
    const FunctionsComparator::Result result = func_comparator.compare(clone, expected.getFunction());
    ASSERT_TRUE(result.valid) << result.message;
}
