// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include "frontend_test.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

class IRFrontendTestsPreProcessing : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendTestsPreProcessing, pre_processing) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="10">
    <pre-process mean-precision="FP32" reference-layer-name="input">
        <channel id="0">
            <mean offset="0" size="1936"/>
        </channel>
        <channel id="1">
            <mean offset="1936" size="1936"/>
        </channel>
        <channel id="2">
            <mean offset="3872" size="1936"/>
        </channel>
    </pre-process>
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    int dataSizeinFloat = 22 * 22 * 3;
    std::vector<unsigned char> buffer(dataSizeinFloat * sizeof(float), 0);
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    for (int i = 0; i < dataSizeinFloat; i++) {
        floatBuffer[i] = 1;
    }

    createTemporalModelFile(xmlModel, buffer);

    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

namespace ov {
namespace test {

using testing::ElementsAre;
using testing::Property;
using testing::UnorderedElementsAre;

TEST_F(IRFrontendTestsPreProcessing, check_tensor_names_after_read_and_pre_post_processing) {
    std::string xml_model = R"V0G0N(
<?xml version="1.0" ?>
<net name="Model" version="11">
	<layers>
		<layer id="0" name="A" type="Parameter" version="opset1">
			<data shape="" element_type="f32" />
			<output>
				<port id="0" precision="f32" names="input_a" />
			</output>
		</layer>
        <layer id="1" name="B" type="Parameter" version="opset1">
			<data shape="" element_type="f32" />
			<output>
				<port id="0" precision="f32" names="input_b" />
			</output>
		</layer>
		<layer id="2" name="my_const" type="Const" version="opset1">
			<data element_type="f32" shape="" offset="0" size="4" />
			<output>
				<port id="0" precision="f32" />
			</output>
		</layer>
		<layer id="3" name="Add" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="f32" />
				<port id="1" precision="f32" />
			</input>
			<output>
				<port id="0" precision="f32" names="add_result" />
			</output>
		</layer>
		<layer id="4" name="output_a" type="Result" version="opset1">
			<input>
				<port id="0" precision="f32" />
			</input>
		</layer>
        <layer id="5" name="output_b" type="Result" version="opset1">
			<input>
				<port id="0" precision="f32" />
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="5" to-port="0" />
        <edge from-layer="2" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="3" from-port="0" to-layer="4" to-port="0" />
	</edges>
	<rt_info />
</net>
// )V0G0N";

    constexpr auto DATA_COUNT = 1;
    std::vector<unsigned char> buffer(DATA_COUNT * sizeof(float), 0);
    std::fill_n(reinterpret_cast<float*>(buffer.data()), DATA_COUNT, 1.f);

    createTemporalModelFile(xml_model, buffer);

    std::shared_ptr<Model> model;
    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_NE(model, nullptr);

    EXPECT_THAT(model->inputs(),
                ElementsAre(Property("Input 0", &Output<Node>::get_names, UnorderedElementsAre("input_a")),
                            Property("Input 1", &Output<Node>::get_names, UnorderedElementsAre("input_b"))));

    EXPECT_THAT(model->outputs(),
                ElementsAre(Property("Output 0", &Output<Node>::get_names, UnorderedElementsAre("add_result")),
                            // Directly connected to model input shows input's names.
                            Property("Output 1", &Output<Node>::get_names, UnorderedElementsAre("input_b"))));

    auto p = preprocess::PrePostProcessor(model);
    p.output(0).tensor().set_element_type(element::f16);
    p.output(1).tensor().set_element_type(element::i32);
    model = p.build();

    EXPECT_THAT(model->inputs(),
                ElementsAre(Property("Input 0", &Output<Node>::get_names, UnorderedElementsAre("input_a")),
                            Property("Input 1", &Output<Node>::get_names, UnorderedElementsAre("input_b"))));

    EXPECT_THAT(model->outputs(),
                ElementsAre(Property("Output 0", &Output<Node>::get_names, UnorderedElementsAre("add_result")),
                            // After PPP (inserts convert node) the tensor names stay on model's input.
                            Property("Output 1", &Output<Node>::get_names, testing::IsEmpty())));
}
}  // namespace test
}  // namespace ov
