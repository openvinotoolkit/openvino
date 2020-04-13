// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
/*
TEST_F(NGraphReaderTests, ReadNonMaxSuppression) {
    std::string model = R"V0G0N(
<net name="NonMaxSuppression" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" >
            <data element_type="f32" shape="1,15130,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" >
            <data element_type="f32" shape="1,80,15130"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="max_output_boxes_per_class" precision="I64" type="Const">
            <data offset="0" size="8"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="3" name="iou_threshold" precision="FP32" type="Const">
            <data offset="8" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="4" name="score_threshold" precision="FP32" type="Const">
            <data offset="12" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="5" name="nms" type="NonMaxSuppression">
            <data box_encoding="corner" sort_result_descending="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
                <port id="2"/>
                <port id="3"/>
                <port id="4"/>
            </input>
            <output>
                <port id="5" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="mul" type="Multiply">
            <input>
                <port id="0">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="output" type="Result" >
            <input>
                <port id="0" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="1"/>
        <edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="NonMaxSuppression" version="5">
    <layers>
        <layer id="0" name="in1" type="Input" >
            <data precision="I32"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" >
            <data precision="I32"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="max_output_boxes_per_class" precision="I64" type="Const">
            <data offset="0" size="8"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="3" name="iou_threshold" precision="FP32" type="Const">
            <data offset="8" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="4" name="score_threshold" precision="FP32" type="Const">
            <data offset="12" size="4"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="5" name="nms" type="NonMaxSuppression" precision="I32">
            <data box_encoding="corner" sort_result_descending="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
                <port id="2"/>
                <port id="3"/>
                <port id="4"/>
            </input>
            <output>
                <port id="5">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="mul" type="Multiply" precision="I32">
            <input>
                <port id="0">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 16, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 200;

        auto * fp32w = weights->buffer().as<float*>();
        fp32w[2] = 0.5;
        fp32w[3] = 0.05;
    });
}

 */