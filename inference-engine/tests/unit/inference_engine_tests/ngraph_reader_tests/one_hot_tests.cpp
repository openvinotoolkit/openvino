// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

TEST_F(NGraphReaderTests, ReadOneHotFP32) {
    std::string model = R"V0G0N(
<net name="OneHot" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
            <data element_type="i64" shape="1,10,22"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="2" name="data2" precision="FP32" type="Const" version="opset1">
            <data offset="8" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="3" name="data3" precision="FP32" type="Const" version="opset1">
            <data offset="12" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="4" name="onehot" type="OneHot" version="opset1">
            <data axis="1"/>
            <input>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
                <port id="1"/>
                <port id="2"/>
                <port id="3"/>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="output" type="Result"  version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="OneHot" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="I64" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="onehot" type="OneHot" precision="FP32">
            <data axis="1" on_value="1.25" off_value="-4.0" depth="5"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 16, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 5;

        auto * fp32w = weights->buffer().as<float*>();
        fp32w[2] = 1.25;
        fp32w[3] = -4.0;
    });
}

TEST_F(NGraphReaderTests, ReadOneHotINT16) {
    std::string model = R"V0G0N(
<net name="OneHot" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
            <data element_type="i64" shape="1,10,22"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="2" name="data2" precision="I16" type="Const" version="opset1">
            <data offset="8" size="2"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="3" name="data3" precision="I16" type="Const" version="opset1">
            <data offset="10" size="2"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="4" name="one_hot_v10" type="OneHot" version="opset1">
            <data axis="1"/>
            <input>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
                <port id="1"/>
                <port id="2"/>
                <port id="3"/>
            </input>
            <output>
                <port id="4" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="mul" type="Multiply" version="opset1">
            <input>
                <port id="0" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result"  version="opset1">
            <input>
                <port id="0" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="1"/>
        <edge from-layer="5" from-port="2" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="OneHot" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="I64" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="one_hot_v5" type="OneHot">
            <data axis="1" on_value="-4" off_value="7" depth="5"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="onehot/Convert" type="Convert" precision="I16">
            <data precision="I16"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Eltwise" precision="I16">
            <data operation="prod"/>
            <input>
                <port id="0" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I16">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>10</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 12, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 5;

        auto * i16w = weights->buffer().as<int16_t*>();
        i16w[4] = -4;
        i16w[5] = 7;
    });
}
