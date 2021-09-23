// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <inference_engine.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

using namespace ngraph;

TEST(RTInfoDeserialization, Node) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,3,22,22"/>
            <rt_info>
                <fused_names:0 value="in1"/>
            </rt_info>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
		<layer name="Round" id="1" type="Round" version="opset8">
			<data mode="half_to_even"/>
            <rt_info>
                <fused_names:0 value="Round1,Round2"/>
            </rt_info>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
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
    auto core = InferenceEngine::Core();
    auto net = core.ReadNetwork(model, InferenceEngine::Blob::Ptr());
    auto f = net.getFunction();

    auto check_fused_names = [](const RTMap & info, const std::string & names) {
        const std::string & key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        ASSERT_EQ(fused_names_attr->get().getNames(), names);
    };

    auto param = f->get_parameters()[0];
    check_fused_names(param->get_rt_info(), "in1");

    auto result = f->get_results()[0];
    auto round = result->get_input_node_ptr(0);
    check_fused_names(round->get_rt_info(), "Round1,Round2");
}

TEST(RTInfoDeserialization, InputAndOutput) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <rt_info>
                        <fused_names:0 value="test1,test2"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="sum" type="Add" version="opset1">
			<input>
				<port id="0">
                    <rt_info>
                        <fused_names:0 value="test2,test3"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
				</port>
				<port id="1">
                    <rt_info>
                        <fused_names:0 value="test3,test4"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
                    <rt_info>
                        <fused_names:0 value="test4,test5"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
				</port>
			</output>
		</layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP32">
                    <rt_info>
                        <fused_names:0 value="test5,test6"/>
                    </rt_info>
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    auto core = InferenceEngine::Core();
    auto net = core.ReadNetwork(model, InferenceEngine::Blob::Ptr());
    auto f = net.getFunction();

    auto check_fused_names = [](const RTMap & info, const std::string & names) {
        const std::string & key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        ASSERT_EQ(fused_names_attr->get().getNames(), names);
    };

    auto param = f->get_parameters()[0];
    check_fused_names(param->output(0).get_rt_info(), "test1,test2");

    auto result = f->get_results()[0];
    check_fused_names(result->input(0).get_rt_info(), "test5,test6");

    auto add = result->get_input_node_ptr(0);
    check_fused_names(add->input(0).get_rt_info(), "test2,test3");
    check_fused_names(add->input(1).get_rt_info(), "test3,test4");
    check_fused_names(add->output(0).get_rt_info(), "test4,test5");
}
