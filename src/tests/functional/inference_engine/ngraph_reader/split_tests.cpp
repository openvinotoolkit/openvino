// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadSplitNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
            <data element_type="f32" shape="1,6,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="" size="8"/>
            <output>
                <port id="0" precision="I64"/>
            </output>
        </layer>
        <layer id="2" name="split" type="Split" version="opset1">
            <data num_splits="2"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64"/>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output1" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output2" type="Result" id="4" version="opset1">
            <input>
                <port id="0" precision="FP32">
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
        <edge from-layer="2" from-port="3" to-layer="4" to-port="0"/>
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
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 8, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
    });
}

TEST_F(NGraphReaderTests, ReadSplitNetwork2) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,63,46,46"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>63</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="" size="8"/>
            <output>
                <port id="0" precision="I64"/>
            </output>
        </layer>
        <layer id="274" name="split_res5c_branch1a_sqr239" type="Split" version="opset1">
            <data num_splits="3"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>63</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="1" precision="I64"/>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer id="275" name="res5c_bone_length_sqr/sum_1" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer id="276" name="res5c_bone_length_sqr/sum_2" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="output1" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="274" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="274" to-port="1"/>
        <edge from-layer="276" from-port="2" to-layer="5" to-port="0"/>
        <edge from-layer="274" from-port="2" to-layer="275" to-port="0"/>
        <edge from-layer="274" from-port="3" to-layer="275" to-port="1"/>
        <edge from-layer="275" from-port="2" to-layer="276" to-port="0"/>
        <edge from-layer="274" from-port="4" to-layer="276" to-port="1"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>63</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="split_res5c_branch1a_sqr239" type="Split" precision="FP32" id="1">
            <data axis="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>63</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="res5c_bone_length_sqr/sum_1" type="Eltwise" precision="FP32" id="2">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="res5c_bone_length_sqr/sum_2" type="Eltwise" precision="FP32" id="3">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>21</dim>
                    <dim>46</dim>
                    <dim>46</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1" />
        <edge from-layer="1" from-port="3" to-layer="3" to-port="1" />
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 8, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
    });
}

TEST_F(NGraphReaderTests, ReadVariadicSplitNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter"  version="opset1">
            <data element_type="f32" shape="1,6,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="" size="8"/>
            <output>
                <port id="0" precision="I64"/>
            </output>
        </layer>
        <layer id="2" name="const2" type="Const" version="opset1">
            <data element_type="i64" offset="8" shape="2" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="split" type="VariadicSplit" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64"/>
                <port id="2" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output1" type="Result" id="4" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output2" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
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
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 24, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 2;
        data[2] = 4;
    });
}
