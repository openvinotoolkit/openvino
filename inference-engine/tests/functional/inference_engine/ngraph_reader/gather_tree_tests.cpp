// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadGatherTreeNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
            <data element_type="f32" shape="100,1,10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter"  version="opset1">
            <data element_type="f32" shape="100,1,10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Parameter"  version="opset1">
            <data element_type="f32" shape="1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="const1" type="Const" version="opset1">
            <data offset="0" size="4"/>
            <output>
                <port id="0" precision="FP32">
                </port>
            </output>
        </layer>
        <layer id="4" type="GatherTree" name="gather_tree" version="opset1">
            <input>
                <port id="0">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                </port>
                <port id="3">
                </port>
            </input>
            <output>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="output1" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" >
            <output>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" >
            <output>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Input" >
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="const1" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="4"/>
            </blobs>
        </layer>
        <layer id="4" type="GatherTree" name="gather_tree">
            <input>
                <port id="0">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="0">
                    <dim>100</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 10);
}
