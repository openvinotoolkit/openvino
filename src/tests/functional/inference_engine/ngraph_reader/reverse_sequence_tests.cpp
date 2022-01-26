// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadReverseSequenceNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="3,10,100,200"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="3" size="24"/>
            <output>
                <port id="1" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reverse_sequence" type="ReverseSequence" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24"/>
            </blobs>
        </layer>
        <layer id="3" name="reverse_sequence" precision="FP32" type="ReverseSequence">
            <data batch_axis="0" seq_axis="1"/>
            <input>
                <port id="0">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>3</dim>
                    <dim>10</dim>
                    <dim>100</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 24, nullptr);
}
