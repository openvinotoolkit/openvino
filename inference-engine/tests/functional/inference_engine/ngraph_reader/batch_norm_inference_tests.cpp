// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, DISABLED_ReadBatchNormInferenceNetwork) {
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
        <layer id="11" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="36" />
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="12" name="conv" type="Convolution" version="opset1">
            <data  dilations="1,1" group="1" kernel="1,1" output="0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
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
        <layer id="1" name="a" type="Const" version="opset1">
            <data offset="0" size="12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="a1" type="Const" version="opset1">
            <data offset="12" size="12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="a2" type="Const" version="opset1">
            <data offset="24" size="12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="a3" type="Const" version="opset1">
            <data offset="36" size="12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="bn" id="5" type="BatchNormInference" version="opset1">
            <data eps="0.1" />
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>3</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>3</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>3</dim>
                </port>
                <port id="5" precision="FP32">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="6" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="12" to-port="0"/>
        <edge from-layer="11" from-port="0" to-layer="12" to-port="1"/>
        <edge from-layer="12" from-port="2" to-layer="5" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="5"/>
        <edge from-layer="5" from-port="6" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
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
        <layer id="3" name="bn" precision="FP32" type="Convolution">
            <data dilations="1,1" group="1" kernel="1,1" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" originalLayersNames="bn,conv"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
            <weights offset="0" size="36" />
            <biases offset="0" size="12"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840);
}
