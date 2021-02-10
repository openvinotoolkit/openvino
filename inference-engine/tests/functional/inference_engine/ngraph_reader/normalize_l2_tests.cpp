// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadNormalizeL2Network) {
    std::string model = R"V0G0N(
<net name="saved_model" version="10">
    <layers>
        <layer id="0" name="input_a" type="Parameter" version="opset1">
            <data shape="6,24,12,10" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="112_input_port_1/value114_const" type="Const" version="opset1">
            <data offset="0" size="8" shape="1" element_type="i64"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="112" type="NormalizeL2" version="opset1">
            <data eps="1e-12" eps_mode="add"/>
            <input>
                <port id="0">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="5354_const" type="Const" version="opset1">
            <data offset="8" size="4" shape="1" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="PartitionedCall/functional_1/lambda/output" type="Multiply" version="opset1">
            <input>
                <port id="0">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Identity/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="saved_model" version="7">
    <layers>
        <layer id="0" name="input_a" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="PartitionedCall/functional_1/lambda/output" type="Normalize">
            <data eps="1e-12" across_spatial="0" channel_shared="1"/>
            <input>
                <port id="0">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>6</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                    <dim>10</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="96" precision="FP32"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 100, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 1;
                buffer[1] = 32831;
             });
}
