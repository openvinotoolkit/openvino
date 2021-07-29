// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadLrnNetwork) {
    std::string model_v10 = R"V0G0N(
<net name="LRN" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const" version="opset1">
            <data offset="0" size="16" shape="2" element_type="i64"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="activation" type="LRN" version="opset2">
            <data alpha="0" beta="0.75" size="5" bias="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
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

    std::string model_v5 = R"V0G0N(
<net name="LRN" version="6">
    <layers>
        <layer id="0" name="in1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="activation" precision="FP32" type="Norm">
            <data alpha="0.000000" beta="0.750000" k="1" local-size="5" region="same" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</net>
)V0G0N";

    compareIRs(model_v10, model_v5, 16, [](Blob::Ptr& weights) {
                auto * w = weights->buffer().as<int64_t*>();
                w[0] = 2;
                w[1] = 3;
            });
}

TEST_F(NGraphReaderTests, ReadLrnNetwork2) {
    std::string model_v10 = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const" version="opset1">
            <data offset="0" size="8" shape="1" element_type="i64"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="activation" type="LRN" version="opset2">
            <data alpha="0" beta="0.75" size="5" bias="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
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

    std::string model_v5 = R"V0G0N(
<net name="Activation" version="6">
    <layers>
        <layer id="0" name="in1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="activation" precision="FP32" type="Norm">
            <data alpha="0.000000" beta="0.750000" k="1" local-size="5" region="across" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</net>
)V0G0N";

    compareIRs(model_v10, model_v5, 8, [](Blob::Ptr& weights) {
        auto * w = weights->buffer().as<int64_t*>();
        w[0] = 1;
    });
}
