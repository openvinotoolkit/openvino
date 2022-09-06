// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "openvino/openvino.hpp"

class MetaData : public ::testing::Test {
public:
    ov::Core core;

    std::string ir_with_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
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
                <port id="2" precision="FP32">
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
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <meta_data>
        <MO_version value="TestVersion"/>
        <Runtime_version value="TestVersion"/>
        <cli_parameters>
            <input_shape value="[1, 3, 22, 22]"/>
            <transform value=""/>
            <use_new_frontend value="False"/>
        </cli_parameters>
    </meta_data>
</net>
)V0G0N";

    std::string ir_without_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
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
                <port id="2" precision="FP32">
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
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    void SetUp() override {}
};

TEST_F(MetaData, get_meta_data_from_model_without_info) {
    ov::Core core;
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    EXPECT_EQ(rt_info.find("meta_data"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map_from_model_without_info) {
    ov::Core core;
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    auto it = rt_info.find("meta_data");
    EXPECT_EQ(it, rt_info.end());
    ov::AnyMap meta;
    EXPECT_NO_THROW(meta = model->get_meta_data());
    EXPECT_TRUE(meta.empty());
}

TEST_F(MetaData, get_meta_data) {
    ov::Core core;
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    EXPECT_NE(rt_info.find("meta_data"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map) {
    ov::Core core;
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    ov::AnyMap meta;
    EXPECT_NO_THROW(meta = model->get_meta_data());
    EXPECT_TRUE(!meta.empty());
    auto it = meta.find("MO_version");
    EXPECT_NE(it, meta.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    it = meta.find("Runtime_version");
    EXPECT_NE(it, meta.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    auto it_cli = meta.find("cli_parameters");
    EXPECT_NE(it_cli, meta.end());
    EXPECT_TRUE(it_cli->second.is<ov::AnyMap>());

    auto cli_map = it_cli->second.as<ov::AnyMap>();
    it = cli_map.find("input_shape");
    EXPECT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "[1, 3, 22, 22]");

    it = cli_map.find("transform");
    EXPECT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "");

    it = cli_map.find("use_new_frontend");
    EXPECT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "False");
}
