// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "ie_iextension.h"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"

class CustomOpsSerializationTest : public ::testing::Test {
protected:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
        ov::shutdown();
    }
};

class TemplateOpExtension : public ov::op::Op {
public:
    OPENVINO_OP("Template", "custom_opset");
    TemplateOpExtension() = default;

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

TEST_F(CustomOpsSerializationTest, CustomOpNoExtensions) {
    const std::string model = R"V0G0N(
<net name="Network" version="11">
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
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    ov::Core core;
    auto extension = std::make_shared<ov::OpExtension<TemplateOpExtension>>();
    core.add_extension(extension);
    auto expected = core.read_model(model, ov::Tensor());
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path, ov::pass::Serialize::Version::IR_V11);
    manager.run_passes(expected);
    auto result = core.read_model(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result, expected, true, false, false, true, true);

    ASSERT_TRUE(success) << message;
}
