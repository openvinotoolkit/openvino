// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadReshapeNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048,1,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reshape1" type="Reshape" version="opset1">
            <data special_zero="True"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
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
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="16"/>
            </blobs>
        </layer>
        <layer id="3" name="reshape1" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2048</dim>
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
    compareIRs(model, modelV5, 16, [](Blob::Ptr& weights) {
                auto *data = weights->buffer().as<int64_t *>();
                data[0] = 1;
                data[1] = 2048;
            });
}
