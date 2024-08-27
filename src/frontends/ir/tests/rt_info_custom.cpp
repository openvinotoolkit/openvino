// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"

TEST(RTInfoCustom, basic_RENAMEME) {
    std::string ir = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="27"/>
            <rt_info>
                <custom name="infoA" value="A"/>
                <custom name="infoB" value="B"/>
                <custom name="infoB" value="BB"/>
                <custom name="fused_names_0" value="a_name"/>
                <attribute name="fused_names" version="0" value="the_name"/>
            </rt_info>
            <output>
                <port id="0" precision="FP32" names="input_tensor">
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="opset8">
            <rt_info>
                <custom name="infoC" value="C"/>
                <attribute name="fused_names" version="0" value=""/>
            </rt_info>
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="output_tensor">
                    <rt_info>
                        <custom name="infoD" value="D"/>
                    </rt_info>
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <rt_info>
                <attribute name="primitives_priority" version="0" value="the_prior"/>
                <custom name="primitives_priority_0" value="a_prior"/>
            </rt_info>
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
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
    auto model = core.read_model(ir, ov::Tensor());
    ASSERT_NE(nullptr, model);
    std::string value;

    const auto& param_rti = model->get_parameters().at(0)->get_rt_info();
    EXPECT_EQ(param_rti.size(), 3);

    OV_ASSERT_NO_THROW(value = param_rti.at("fused_names_0").as<std::string>());
    EXPECT_EQ(value.compare("the_name"), 0);

    OV_ASSERT_NO_THROW(value = param_rti.at("infoA").as<std::string>());
    EXPECT_EQ(value.compare("A"), 0);

    OV_ASSERT_NO_THROW(value = param_rti.at("infoB").as<std::string>());
    EXPECT_EQ(value.compare("B"), 0);

    const auto& result = model->get_results().at(0);
    const auto abs = result->get_input_node_ptr(0);

    const auto& abs_rti = abs->get_rt_info();
    EXPECT_EQ(abs_rti.size(), 2);
    OV_ASSERT_NO_THROW(value = abs_rti.at("infoC").as<std::string>());
    EXPECT_EQ(value.compare("C"), 0);

    const auto& abs_output_rti = abs->output(0).get_rt_info();
    EXPECT_EQ(abs_output_rti.size(), 1);
    OV_ASSERT_NO_THROW(value = abs_output_rti.at("infoD").as<std::string>());
    EXPECT_EQ(value.compare("D"), 0);

    const auto& result_rti = result->get_rt_info();
    EXPECT_EQ(result_rti.size(), 1);
    OV_ASSERT_NO_THROW(value = result_rti.at("primitives_priority_0").as<std::string>());
    EXPECT_EQ(value.compare("the_prior"), 0);
}
