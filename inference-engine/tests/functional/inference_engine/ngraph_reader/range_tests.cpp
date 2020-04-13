// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadRangeNetwork) {
    std::string model = R"V0G0N(
<net name="Range" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2,12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="start" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0" precision="I64">
                </port>
            </output>
        </layer>
        <layer id="5" name="stop" type="Const" version="opset1">
            <data offset="8" size="8"/>
            <output>
                <port id="0" precision="I64">
                </port>
            </output>
        </layer>
        <layer id="6" name="step" type="Const" version="opset1">
            <data offset="16" size="8"/>
            <output>
                <port id="0" precision="I64">
                </port>
            </output>
        </layer>

        <layer id="3" name="range"  type="Range" version="opset1">
            <input>
                <port id="0" precision="I64">
                </port>
                <port id="1" precision="I64">
                </port>
                <port id="2" precision="I64">
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>

        <layer id="1" name="reshape"  type="Reshape" version="opset1">
            <data special_zero="True"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="output" type="Result"  version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="5" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="6" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Range" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="3" name="reshape" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
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

    compareIRs(model, modelV5, 80, [](Blob::Ptr& weights) {
        auto * w = weights->buffer().as<int64_t*>();
        w[0] = 1;
        w[1] = 5;
        w[2] = 1;
    });
}
