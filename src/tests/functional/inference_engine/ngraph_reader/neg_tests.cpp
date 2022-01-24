// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadNegativeNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
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
        <layer id="1" name="neg" type="Negative" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
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
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="neg" precision="FP32" type="Power">
             <data power="1.000000" scale="-1.000000" shift="0.000000"/>
            <input>
                <port id="0">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 0);
}
