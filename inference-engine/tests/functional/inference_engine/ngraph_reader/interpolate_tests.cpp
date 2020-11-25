// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadInterpolateNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" precision="FP32" version="opset1">
            <data element_type="f32" shape="1,2,48,80"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" precision="I64" version="opset1">
            <data element_type="i64" offset="0" shape="2" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="interpolate" type="Interpolate" precision="FP32" version="opset1">
            <data axes="2,3" align_corners="0" pads_begin="0" pads_end="0" mode="linear"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" precision="FP32" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="16"/>
            </blobs>
        </layer>
        <layer id="2" name="scales" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="32" size="16"/>
            </blobs>
        </layer>
        <layer id="3" name="axes" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="16" size="16"/>
            </blobs>
        </layer>
        <layer id="4" name="interpolate" precision="FP32" type="Interpolate">
            <data antialias="False" coordinate_transformation_mode="half_pixel" cube_coeff="-0.75" mode="linear" nearest_mode="round_prefer_floor" pads_begin="0,0,0,0" pads_end="0,0,0,0" shape_calculation_mode="sizes"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
                <port id="3">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 48, [](Blob::Ptr& weights) {
                auto *i64data = weights->buffer().as<int64_t *>();
                i64data[0] = 50;
                i64data[1] = 60;
                i64data[2] = 2;
                i64data[3] = 3;

                auto *fdata = reinterpret_cast<float *>(i64data + 4);
                fdata[0] = 1.0;
                fdata[1] = 1.0;
                fdata[2] = 50.0 / 48.0;
                fdata[3] = 60.0 / 80.0;
            });
}

TEST_F(NGraphReaderTests, ReadInterpolate2Network) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" precision="FP32" version="opset1">
            <data element_type="f32" shape="1,2,48,80"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" precision="I64" version="opset1">
            <data element_type="i64" offset="0" shape="4" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="interpolate" type="Interpolate" precision="FP32" version="opset1">
            <data axes="0,2,1,3" align_corners="0" pads_begin="0" pads_end="0" mode="linear"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" precision="FP32" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="scales" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="64" size="16"/>
            </blobs>
        </layer>
        <layer id="3" name="axes" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="32" size="32"/>
            </blobs>
        </layer>
        <layer id="4" name="interpolate" precision="FP32" type="Interpolate">
            <data antialias="False" coordinate_transformation_mode="half_pixel" cube_coeff="-0.75" mode="linear" nearest_mode="round_prefer_floor" pads_begin="0,0,0,0" pads_end="0,0,0,0" shape_calculation_mode="sizes"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
                <port id="2">
                    <dim>4</dim>
                </port>
                <port id="3">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 80, [](Blob::Ptr& weights) {
                auto *i64data = weights->buffer().as<int64_t *>();
                i64data[0] = 1;
                i64data[1] = 2;
                i64data[2] = 50;
                i64data[3] = 60;
                i64data[4] = 0;
                i64data[5] = 1;
                i64data[6] = 2;
                i64data[7] = 3;

                auto *fdata = reinterpret_cast<float *>(i64data + 8);
                fdata[0] = 1.0;
                fdata[1] = 1.0;
                fdata[2] = 50.0 / 48.0;
                fdata[3] = 60.0 / 80.0;
            });
}

TEST_F(NGraphReaderTests, ReadInterpolate4Network) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" precision="FP32" version="opset1">
            <data element_type="f32" shape="1,2,300,300"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output_shape" type="Const" precision="I32" version="opset1">
            <data element_type="i32" offset="0" shape="2" size="8"/>
            <output>
                <port id="1" precision="I32">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="scales" type="Const" precision="FP32" version="opset1">
            <data element_type="f32" offset="8" shape="2" size="8"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="axes" type="Const" precision="I32" version="opset1">
            <data element_type="i32" offset="16" shape="2" size="8"/>
            <output>
                <port id="1" precision="I32">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="interpolate" type="Interpolate" precision="FP32" version="opset4">
            <data antialias="0" coordinate_transformation_mode="asymmetric" cube_coeff="123" mode="nearest" nearest_mode="floor" pads_begin="2,3,4,5" pads_end="6,7,8,9" shape_calculation_mode="sizes"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>2</dim>
                </port>
                <port id="3">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>9</dim>
                    <dim>12</dim>
                    <dim>600</dim>
                    <dim>900</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="5" precision="FP32" version="opset1">
            <input>
                <port id="0">
                    <dim>9</dim>
                    <dim>12</dim>
                    <dim>600</dim>
                    <dim>900</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output_shape" precision="I32" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="8"/>
            </blobs>
        </layer>
        <layer id="2" name="scales" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="8" size="8"/>
            </blobs>
        </layer>
        <layer id="3" name="axes" precision="I32" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="16" size="8"/>
            </blobs>
        </layer>
        <layer id="4" name="interpolate" precision="FP32" type="Interpolate">
            <data antialias="False" coordinate_transformation_mode="asymmetric" cube_coeff="123" mode="nearest" nearest_mode="floor" pads_begin="2,3,4,5" pads_end="6,7,8,9" shape_calculation_mode="sizes"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>2</dim>
                </port>
                <port id="3">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>9</dim>
                    <dim>12</dim>
                    <dim>600</dim>
                    <dim>900</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 24, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int*>();
        data[0] = 600;
        data[1] = 900;
        data[4] = 2;
        data[5] = 3;

        auto *fdata = weights->buffer().as<float*>();
        fdata[2] = 2.0;
        fdata[3] = 2.0;
    });
}
