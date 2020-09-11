// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadUnsqueeze) {
    std::string model_version10 = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="32,64,60"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ExpandDims" precision="FP32" type="Unsqueeze" version="opset1">
            <input>
                <port id="0">
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
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
    std::string model_version6 = R"V0G0N(
<net name="Network" version="6" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer name="const1" type="Const" precision="I64" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="8" />
            </blobs>
        </layer>
        <layer name="ExpandDims" type="Unsqueeze" precision="FP32" id="2">
            <input>
                <port id="0">
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>32</dim>
                    <dim>64</dim>
                    <dim>60</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
    </edges>
    <statistics />
</net>
)V0G0N";

    compareIRs(model_version10, model_version6, 8, [](Blob::Ptr& weights) {
                auto* w = weights->buffer().as<int64_t*>();
                w[0] = 3;
            });
}
