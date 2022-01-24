// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include "common_test_utils/data_utils.hpp"

TEST_F(NGraphReaderTests, ReadIDFTNetwork) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data shape="1,180,180,2" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <data offset="0" size="16" shape="2" element_type="i64"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
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
    std::string modelV7 = R"V0G0N(
<net name="deformable_convolution" version="7">
    <layers>
        <layer id="0" name="in1" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="16" precision="I64"/>
            </blobs>
        </layer>
        <layer id="2" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>180</dim>
                    <dim>180</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV7, 16, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 2;
        i64w[1] = 0;
    });
}

TEST_F(NGraphReaderTests, ReadIDFTNetwork2) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data shape="7,50,130,400,2" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>50</dim>
                    <dim>130</dim>
                    <dim>400</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <data offset="0" size="24" shape="3" element_type="i64"/>
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="signal_size" type="Const" version="opset1">
            <data offset="24" size="24" shape="3" element_type="i64"/>
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>50</dim>
                    <dim>130</dim>
                    <dim>400</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
                <port id="2">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>7</dim>
                    <dim>40</dim>
                    <dim>130</dim>
                    <dim>600</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>40</dim>
                    <dim>130</dim>
                    <dim>600</dim>
                    <dim>2</dim>
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
    std::string modelV7 = R"V0G0N(
<net name="deformable_convolution" version="7">
    <layers>
        <layer id="0" name="in1" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>50</dim>
                    <dim>130</dim>
                    <dim>400</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24" precision="I64"/>
            </blobs>
        </layer>
        <layer id="2" name="signal_size" type="Const" version="opset1">
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="24" size="24" precision="I64"/>
            </blobs>
        </layer>
        <layer id="3" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>50</dim>
                    <dim>130</dim>
                    <dim>400</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
                <port id="2">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>7</dim>
                    <dim>40</dim>
                    <dim>130</dim>
                    <dim>600</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV7, 48, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 3;
        i64w[1] = 0;
        i64w[2] = 1;
        i64w[3] = 600;
        i64w[4] = -1;
        i64w[5] = 40;
    });
}

TEST_F(NGraphReaderTests, ReadIDFTNetwork3) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data shape="7,15,200,124,70,2" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>200</dim>
                    <dim>124</dim>
                    <dim>70</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <data offset="0" size="24" shape="3" element_type="i64"/>
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="signal_size" type="Const" version="opset1">
            <data offset="24" size="24" shape="3" element_type="i64"/>
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>200</dim>
                    <dim>124</dim>
                    <dim>70</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
                <port id="2">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>100</dim>
                    <dim>124</dim>
                    <dim>280</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>100</dim>
                    <dim>124</dim>
                    <dim>280</dim>
                    <dim>2</dim>
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
    std::string modelV7 = R"V0G0N(
<net name="deformable_convolution" version="7">
    <layers>
        <layer id="0" name="in1" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>200</dim>
                    <dim>124</dim>
                    <dim>70</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="axes" type="Const" version="opset1">
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24" precision="I64"/>
            </blobs>
        </layer>
        <layer id="2" name="signal_size" type="Const" version="opset1">
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="24" size="24" precision="I64"/>
            </blobs>
        </layer>
        <layer id="3" name="dft" type="IDFT" version="opset7">
            <input>
                <port id="0">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>200</dim>
                    <dim>124</dim>
                    <dim>70</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
                <port id="2">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>7</dim>
                    <dim>15</dim>
                    <dim>100</dim>
                    <dim>124</dim>
                    <dim>280</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV7, 48, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = -3;
        i64w[1] = 4;
        i64w[2] = 0;
        i64w[3] = 100;
        i64w[4] = 280;
        i64w[5] = -1;
    });
}
