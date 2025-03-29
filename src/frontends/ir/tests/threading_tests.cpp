// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "openvino/openvino.hpp"

class IRFRThreadingTests : public ::testing::Test {
public:
    void run_parallel(std::function<void(void)> func,
                      const unsigned int iterations = 100,
                      const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }

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
            <cli_parameters>
                <input_shape value="[1, 3, 22, 22]"/>
                <transform value=""/>
                <use_new_frontend value="False"/>
                <MO_version value="TestVersion"/>
                <Runtime_version value="TestVersion"/>
                <cli_parameters>
                    <input_shape value="[1, 3, 22, 22]"/>
                    <transform value=""/>
                    <use_new_frontend value="False"/>
                    <cli_parameters>
                        <input_shape value="[1, 3, 22, 22]"/>
                        <transform value=""/>
                        <use_new_frontend value="False"/>
                    </cli_parameters>
                </cli_parameters>
            </cli_parameters>
        </cli_parameters>
    </meta_data>
</net>
)V0G0N";

    void SetUp() override {}
};

TEST_F(IRFRThreadingTests, get_meta_data_in_different_threads) {
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    run_parallel([&]() {
        auto& rt_info = model->get_rt_info();
        auto it = rt_info.find("MO_version");
        ASSERT_NE(it, rt_info.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

        it = rt_info.find("Runtime_version");
        ASSERT_NE(it, rt_info.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

        ov::AnyMap cli_map;
        EXPECT_NO_THROW(cli_map = model->get_rt_info<ov::AnyMap>("conversion_parameters"));

        it = cli_map.find("input_shape");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "[1, 3, 22, 22]");

        it = cli_map.find("transform");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "");

        it = cli_map.find("use_new_frontend");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "False");
    });
}
