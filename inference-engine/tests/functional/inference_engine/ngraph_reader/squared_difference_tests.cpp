// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadSquaredDifferenceNetwork) {
    std::string model = R"V0G0N(
<net name="Less" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,64,112,112"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const" version="opset1">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="sq_diff" type="SquaredDifference" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Greater" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="sq_diff" precision="FP32" type="Eltwise">
            <data operation="squared_diff"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 3211264);
}

