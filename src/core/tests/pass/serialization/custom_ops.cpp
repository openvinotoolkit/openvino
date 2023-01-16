// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_iextension.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/serialize.hpp"

class CustomOpsSerializationTest : public ::testing::Test {
protected:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        std::string filePrefix = CommonTestUtils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
        ov::shutdown();
    }
};

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
    const std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Template" version="custom_opset">
            <data num_bodies="0" add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    ov::Core core;
    auto extension = std::make_shared<FrameworkNodeExtension>();
    core.add_extension(extension);
    auto expected = core.read_model(model, ov::Tensor());
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_out_xml_path,
                                               m_out_bin_path,
                                               extension->getOpSets(),
                                               ov::pass::Serialize::Version::IR_V10);
    manager.run_passes(expected);
    auto result = core.read_model(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result, expected, true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}
