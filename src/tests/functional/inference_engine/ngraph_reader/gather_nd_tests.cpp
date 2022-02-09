// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadGatherNDNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="params_x" type="Parameter" version="opset1">
            <data element_type="f32" shape="10,20,30"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                    <dim>20</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="indices_y" type="Parameter" version="opset1">
            <data element_type="i32" shape="10,3,2"/>
            <output>
                <port id="0" precision="I32">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="MyGatherND" type="GatherND" version="opset5">
            <data batch_dims="0"/>
            <input>
                <port id="0">
                    <dim>10</dim>
                    <dim>20</dim>
                    <dim>30</dim>
                </port>
                <port id="1">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="MyGatherND/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>30</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="params_x" type="Input" precision="FP32">
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                    <dim>20</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="indices_y" type="Input" precision="I32">
            <output>
                <port id="0" precision="I32">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="MyGatherND" type="GatherND" version="opset5">
            <data batch_dims="0"/>
            <input>
                <port id="0">
                    <dim>10</dim>
                    <dim>20</dim>
                    <dim>30</dim>
                </port>
                <port id="1">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>10</dim>
                    <dim>3</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 10);
}
