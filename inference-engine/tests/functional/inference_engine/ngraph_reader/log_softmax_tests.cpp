// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadLogSoftmaxNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,1000"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="log_softmax" id="1" type="LogSoftmax" version="opset5">
            <data axis="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string model_ref = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="LogSoftmax/ReduceMax_axis" type="Const">
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="8"/>
            </blobs>
        </layer>
        <layer id="2" name="LogSoftmax/ReduceMax" type="ReduceMax">
            <data keep_dims="True"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="LogSoftmax/Neg1" type="Power">
            <data power="1.0" scale="-1.0" shift="0.0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="LogSoftmax/Sub/first" type="Eltwise">
            <data operation="sum"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="LogSoftmax/Exp" type="Exp">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="LogSoftmax/ReduceSum_axis" type="Const">
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="8" size="8"/>
            </blobs>
        </layer>
        <layer id="7" name="LogSoftmax/ReduceSum" type="ReduceSum">
            <data keep_dims="True"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="LogSoftmax/Log" type="Log">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="LogSoftmax/Neg2" type="Power">
            <data power="1.0" scale="-1.0" shift="0.0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="log_softmax" type="Eltwise">
            <data operation="sum"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
        <edge from-layer="4" from-port="2" to-layer="10" to-port="0"/>
        <edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>
        <edge from-layer="9" from-port="1" to-layer="10" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, model_ref, 16, [](Blob::Ptr& weights) {
         auto* data = reinterpret_cast<int64_t*>(weights->buffer().as<int8_t*>());
         data[0] = 1;
         data[1] = 1;
     });
}
