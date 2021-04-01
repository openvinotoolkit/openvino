// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, DISABLED_ReadShapeOfNetwork) {
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
        <layer name="activation" id="1" type="ShapeOf" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="activation2" id="3" type="Reshape" precision="FP32" version="opset1">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" id="1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Const" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="3" type="Reshape" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 0);
}

TEST_F(NGraphReaderTests, ReadShapeOfFromScalar) {
    // The test checks case when ShapeOf gets a scalar as input and the result tensor has shape [0]. This means an empty
    // tensor which does not have data. There is nothing to do with this tensor so the test model has another ShapeOf
    // producing tensor with shape [1] which is the output of the model.
    std::string model = R"V0G0N(
    <net name="model_10" version="10">
        <layers>
            <layer id="0" name="Placeholder_2" type="Parameter" version="opset1">
                <data shape="1,3,7" element_type="f32"/>
                <output>
                    <port id="0" precision="FP32" names="Placeholder_2:0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>7</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="Placeholder" type="Parameter" version="opset1">
                <data shape="1" element_type="f32"/>
                <output>
                    <port id="0" precision="FP32" names="Placeholder:0">
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="Shape" type="ShapeOf" version="opset3">
                <data output_type="i32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="1" precision="I32" names="Shape:0">
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="Squeeze/Dims/Output_0/Data__const" type="Const" version="opset1">
                <data offset="0" size="8" shape="1" element_type="i64"/>
                <output>
                    <port id="0" precision="I64">
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="Squeeze" type="Squeeze" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="2" precision="I32" names="Squeeze:0"/>
                </output>
            </layer>
            <layer id="5" name="Shape2" type="ShapeOf" version="opset3">
                <data output_type="i32"/>
                <input>
                    <port id="0">
                    </port>
                </input>
                <output>
                    <port id="1" precision="I32" names="Shape2:0">
                        <dim>0</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="Shape0D" type="ShapeOf" version="opset3">
                <data output_type="i32"/>
                <input>
                    <port id="0">
                        <dim>0</dim>
                    </port>
                </input>
                <output>
                    <port id="1" precision="I32" names="Shape0D:0">
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="7" name="Shape0D/sink_port_0" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="2" from-port="1" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
            <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
            <edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
            <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
        </edges>
    </net>
)V0G0N";

    Blob::Ptr blob;
    blob = make_shared_blob<int64_t>(TensorDesc(Precision::I64, {1}, Layout::C));
    blob->allocate();
    auto *data = blob->buffer().as<int64_t *>();
    data[0] = 0;
    Core reader;
    reader.ReadNetwork(model, blob);
}
