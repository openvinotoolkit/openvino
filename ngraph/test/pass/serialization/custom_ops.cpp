// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "backend.hpp"
#include "ie_api.h"
#include "ie_core.hpp"
#include "ie_iextension.h"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "util/graph_comparator.hpp"
#include "util/test_common.hpp"

#ifndef IE_BUILD_POSTFIX  // should be already defined by cmake
#    define IE_BUILD_POSTFIX ""
#endif

static std::string get_extension_path() {
    std::string library_name = ov::util::FileTraits<char>::library_prefix() + "template_extension" + IE_BUILD_POSTFIX +
                               ov::util::FileTraits<char>::dot_symbol + ov::util::FileTraits<char>::library_ext();
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::string exPath = ov::util::path_join(
        {ov::util::get_directory(ngraph::runtime::Backend::get_backend_shared_library_search_directory()),
         library_name});
    NGRAPH_SUPPRESS_DEPRECATED_END
    return exPath;
}

class CustomOpsSerializationTest : public ov::test::TestsCommon {
protected:
    std::string test_name = GetTestName();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(CustomOpsSerializationTest, CustomOpUser_MO) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/custom_op.xml"});

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

#ifdef NGRAPH_ONNX_FRONTEND_ENABLE

TEST_F(CustomOpsSerializationTest, CustomOpUser_ONNXImporter) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/custom_op.onnx"});

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

#endif

TEST_F(CustomOpsSerializationTest, CustomOpTransformation) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/custom_op.xml"});

    InferenceEngine::Core ie;
    auto extension = std::make_shared<InferenceEngine::Extension>(get_extension_path());
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_out_xml_path,
                                               m_out_bin_path,
                                               extension->getOpSets(),
                                               ov::pass::Serialize::Version::IR_V10);
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
        return {{"framework_node_ext", ngraph::OpSet()}};
    }

    void Unload() noexcept override {}
};

TEST_F(CustomOpsSerializationTest, CustomOpNoExtensions) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/custom_op.xml"});

    InferenceEngine::Core ie;
    auto extension = std::make_shared<FrameworkNodeExtension>();
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_out_xml_path,
                                               m_out_bin_path,
                                               extension->getOpSets(),
                                               ov::pass::Serialize::Version::IR_V10);
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}
