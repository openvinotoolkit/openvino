// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/function.hpp>
#include <string>

#include "ngraph_reader_tests.hpp"
#include "openvino/openvino.hpp"

TEST_F(NGraphReaderTests, ReadNetworkWithTensorNames) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32" names="input">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="relu\,t, identity_t">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    Core ie;
    Blob::Ptr weights;

    auto network = ie.ReadNetwork(model, weights);
    auto function = network.getFunction();
    auto inputs = network.getInputsInfo();
    auto outputs = network.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    ASSERT_EQ(1, inputs.size());
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(1, function->get_results().size());

    for (const auto& param : function->get_parameters()) {
        ASSERT_TRUE(!param->get_output_tensor(0).get_names().empty());
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_TRUE(inNames.count(network.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_TRUE(!result->get_input_tensor(0).get_names().empty());
        for (const auto& name : result->get_input_tensor(0).get_names())
            ASSERT_TRUE(outNames.count(network.getOVNameForTensor(name)));
    }
    ASSERT_NO_THROW(network.getOVNameForTensor("relu,t"));
}

TEST_F(NGraphReaderTests, ReadModelWithTensorNamesWithSpaces) {
    std::string model_str = R"V0G0N(
            <net name="graph" version="11">
            <layers>
                <layer id="1" name="input1" type="Parameter" version="opset1">
                    <data shape="1,4,512" element_type="f32"/>
                    <output>
                        <port id="0" precision="FP32" names="input1">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="0" name="input2" type="Parameter" version="opset1">
                    <data shape="1,4,512" element_type="f32"/>
                    <output>
                        <port id="0" precision="FP32" names="input2">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="output 0([1 4 512])" type="Add" version="opset1">
                    <data auto_broadcast="numpy"/>
                    <input>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                        <port id="1" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP32" names="output 0([1 4 512])">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="output 0([1 4 512])/sink_port_0" type="Result" version="opset1">
                    <input>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            </edges>
        </net>
        )V0G0N";
    ov::Core core;
    auto model = core.read_model(model_str, ov::Tensor{});
    auto outputs = model->outputs();
    EXPECT_EQ(outputs.size(), 1);
    auto names = outputs.at(0).get_names();
    EXPECT_EQ(names.size(), 1);
    auto it = names.find("output 0([1 4 512])");
    EXPECT_NE(it, names.end());
}

TEST_F(NGraphReaderTests, ReadModelWithTensorNamesAddOutput) {
    std::string model_str = R"V0G0N(
<net name="graph" version="11">
	<layers>
		<layer id="1" name="input1" type="Parameter" version="opset1">
			<data shape="1,4,512" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="0" name="input2" type="Parameter" version="opset1">
			<data shape="1,4,512" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input2">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add 221" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="output add">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="output 0([1 4 512])" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="output 0([1 4 512])">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="output 0([1 4 512])/sink_port_0" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="output 0([1 4 512])/sink_port_0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
	</edges>
</net>)V0G0N";
    ov::Core core;
    std::string tensor_name = "output add";
    auto model = core.read_model(model_str, ov::Tensor{});
    model->add_output(tensor_name);
    auto outputs = model->outputs();
    EXPECT_EQ(outputs.size(), 2);
    auto names = outputs.at(1).get_names();
    EXPECT_EQ(names.size(), 1);
    auto it = names.find(tensor_name);
    EXPECT_NE(it, names.end());
}
