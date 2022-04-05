// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadRollNetwork) {
    std::string model = R"V0G0N(
<net name="roll" version="10">
    <layers>
        <layer id="0" name="Input" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,4,2,3"/>
            <output>
                <port id="0" names="Input:0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Roll/shift" type="Parameter" version="opset1">
            <data element_type="i32" offset="0" shape="4" size="16"/>
            <output>
                <port id="0" names="Roll/shift:0" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Roll/axes" type="Parameter" version="opset1">
            <data element_type="i64" offset="16" shape="4" size="32"/>
            <output>
                <port id="0" names="Roll/axes:0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Roll" type="Roll" version="opset7">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" names="Roll:0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Roll:0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
    )V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="roll" version="7">
    <layers>
        <layer id="0" name="Input" type="Input" version="opset1">
            <data element_type="f32" shape="1,4,2,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Roll/shift" type="Input" version="opset1">
            <data element_type="i32" offset="0" shape="4" size="16"/>
            <output>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Roll/axes" type="Input" version="opset1">
            <data element_type="i64" offset="16" shape="4" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Roll" type="Roll" version="opset7">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
    )V0G0N";
    compareIRs(model, modelV7);
}
