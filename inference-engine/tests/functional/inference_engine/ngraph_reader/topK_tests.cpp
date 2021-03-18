// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, DISABLED_ReadTopKNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="1" size="8"/>
            <output>
                <port id="1" precision="I64" />
            </output>
        </layer>
        <layer name="topk" id="1" type="TopK" version="opset1">
            <data axis="2" mode="max" sort="value"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2"/>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="4" precision="I32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="mul_values" type="Multiply">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul_indices" type="Multiply">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output_values" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output_indices" type="Result" id="6">
            <input>
                <port id="0" precision="I32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="1" to-port="2"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="4" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="4" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
        <edge from-layer="3" from-port="2" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const" precision="I64">
            <data element_type="i64" offset="0" shape="1" size="8"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="topk" id="1" type="TopK" precision="FP32">
            <data axis="2" mode="max" sort="value"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="mul_values" type="Eltwise" precision="FP32">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul_indices" type="Eltwise" precision="I32">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 8, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 5;
    });
}
