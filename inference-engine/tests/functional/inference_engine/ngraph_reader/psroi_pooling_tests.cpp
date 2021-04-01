// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadPSROIPoolingNetwork) {
    std::string model = R"V0G0N(
<net name="PSROIPooling" version="10">
    <layers>
        <layer id="0" name="detector/bbox/ps_roi_pooling/placeholder_port_0" type="Parameter" version="opset1">
            <data shape="1,392,34,62" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>392</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="detector/bbox/ps_roi_pooling/placeholder_port_1" type="Parameter" version="opset1">
            <data shape="200,5" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="detector/bbox/ps_roi_pooling" type="PSROIPooling" version="opset2">
            <data spatial_scale="0.0625" spatial_bins_x="3" spatial_bins_y="3" output_dim="8" group_size="7" mode="average"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>392</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="1">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>200</dim>
                    <dim>8</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="detector/bbox/ps_roi_pooling/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>200</dim>
                    <dim>8</dim>
                    <dim>7</dim>
                    <dim>7</dim>
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
<net name="PSROIPooling" version="7">
    <layers>
        <layer id="0" name="detector/bbox/ps_roi_pooling/placeholder_port_0" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>392</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="detector/bbox/ps_roi_pooling/placeholder_port_1" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="detector/bbox/ps_roi_pooling" type="PSROIPooling" version="opset2">
            <data spatial_scale="0.0625" output_dim="8" spatial_bins_x="3" spatial_bins_y="3" group_size="7" mode="average" no_trans="True" trans_std="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>392</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="1">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>200</dim>
                    <dim>8</dim>
                    <dim>7</dim>
                    <dim>7</dim>
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
    compareIRs(model, modelV7);
}
