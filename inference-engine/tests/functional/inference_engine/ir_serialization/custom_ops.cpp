// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <file_utils.h>
#include <ie_api.h>
#include <ie_iextension.h>
#include <ie_network_reader.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

#ifndef IE_BUILD_POSTFIX  // should be already defined by cmake
#define IE_BUILD_POSTFIX ""
#endif

static std::string get_extension_path() {
    return FileUtils::makePluginLibraryName<char>(
        {}, std::string("template_extension") + IE_BUILD_POSTFIX);
}

class CustomOpsSerializationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(CustomOpsSerializationTest, CustomOpUser_MO) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.xml";

    InferenceEngine::Core ie;
    ie.AddExtension(
        std::make_shared<InferenceEngine::Extension>(
            get_extension_path()));

    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

#ifdef NGRAPH_ONNX_IMPORT_ENABLE

TEST_F(CustomOpsSerializationTest, CustomOpUser_ONNXImporter) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.prototxt";

    InferenceEngine::Core ie;
    ie.AddExtension(
        std::make_shared<InferenceEngine::Extension>(
            get_extension_path()));

    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

#endif

TEST_F(CustomOpsSerializationTest, CustomOpTransformation) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.xml";

    InferenceEngine::Core ie;
    auto extension =
        std::make_shared<InferenceEngine::Extension>(
            get_extension_path());
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
        m_out_xml_path, m_out_bin_path,
        ngraph::pass::Serialize::Version::IR_V10, extension->getOpSets());
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true);

    ASSERT_TRUE(success) << message;
}

class FrameworkNodeExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {
        static InferenceEngine::Version ExtensionDescription = {
                {1, 0},
                "1.0",
                "framework_node_ext"
        };

        versionInfo = &ExtensionDescription;
    }

    void Unload() noexcept override {}
};

TEST_F(CustomOpsSerializationTest, CustomOpNoExtensions) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.xml";

    InferenceEngine::Core ie;
    auto extension = std::make_shared<FrameworkNodeExtension>();
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
            m_out_xml_path, m_out_bin_path,
            ngraph::pass::Serialize::Version::IR_V10, extension->getOpSets());
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
            compare_functions(result.getFunction(), expected.getFunction(), true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}
