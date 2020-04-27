// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadPadNoPadValue) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
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
        <layer id="1" name="data1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data2" precision="I64" type="Const" version="opset1">
            <data offset="32" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="pad" type="Pad" version="opset1">
            <data pad_mode="edge" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
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
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result"  version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
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
        <layer id="1" name="pad" type="Pad" precision="FP32">
            <data pad_mode="edge" pads_begin="0,0,1,1" pads_end="0,0,1,1" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 80, [](Blob::Ptr& weights) {
        auto * w = weights->buffer().as<int64_t*>();
        w[0] = 0;
        w[1] = 0;
        w[2] = 1;
        w[3] = 1;

        w[4] = 0;
        w[5] = 0;
        w[6] = 1;
        w[7] = 1;
    });
}

TEST_F(NGraphReaderTests, ReadPadWithPadValue) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
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
        <layer id="1" name="data1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data2" precision="I64" type="Const" version="opset1">
            <data offset="32" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="data3" precision="FP32" type="Const" version="opset1">
            <data offset="64" size="8"/>
            <output>
                <port id="0" precision="FP32">
                </port>
            </output>
        </layer>
        <layer id="3" name="pad" type="Pad" version="opset1">
            <data pad_mode="constant" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
                <port id="3">
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result"  version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="6" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="0"/>
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
        <layer id="1" name="pad" type="Pad" precision="FP32">
            <data pad_mode="constant" pads_begin="0,0,1,1" pads_end="0,0,1,1" pad_value="127.5"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 80, [](Blob::Ptr& weights) {
        auto * w = weights->buffer().as<int64_t*>();
        w[0] = 0;
        w[1] = 0;
        w[2] = 1;
        w[3] = 1;

        w[4] = 0;
        w[5] = 0;
        w[6] = 1;
        w[7] = 1;

        auto * pad_value = reinterpret_cast<float *>(weights->buffer().as<int8_t *>() + 64);
        pad_value[0] = 127.5;
    });
}
