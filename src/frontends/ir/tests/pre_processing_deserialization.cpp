// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_test.hpp"

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
