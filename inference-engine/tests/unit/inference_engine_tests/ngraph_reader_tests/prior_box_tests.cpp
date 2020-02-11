// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

TEST_F(NGraphReaderTests, ReadPriorBoxClusteredNetwork) {
    std::string model = R"V0G0N(
<net name="PriorBoxClusteredNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,768,30,30"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,512,512"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ch_concat_mixed_7_chconcat_anchors/0_port" type="ShapeOf" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="1344813449_const" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const" version="opset1">
            <data offset="8" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="13458134591_const" type="Const" version="opset1">
            <data offset="16" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="ch_concat_mixed_7_chconcat_anchors/ss_0_port" type="StridedSlice" version="opset1">
            <data begin_mask="0" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="ch_concat_mixed_7_chconcat_anchors/1_port" type="ShapeOf" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="ch_concat_mixed_7_chconcat_anchors/ss_1_port" type="StridedSlice" version="opset1">
            <data begin_mask="0" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="prior" type="PriorBoxClustered" version="opset1">
            <data clip="0" flip="0" height="44.0,10.0,30.0,19.0,94.0,32.0,61.0,53.0,17.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2"
                width="86.0,13.0,57.0,39.0,68.0,34.0,142.0,50.0,23.0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </output>
        </layer>
        <layer id="12" name="ExpandAxis" type="Const" version="opset1">
            <data offset="24" size="8"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="11" name="ExpandDims" precision="FP32" type="Unsqueeze" version="opset1">
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="9" from-port="1" to-layer="5" to-port="3"/>
        <edge from-layer="9" from-port="1" to-layer="7" to-port="3"/>
        <edge from-layer="5" from-port="4" to-layer="8" to-port="0"/>
        <edge from-layer="7" from-port="4" to-layer="8" to-port="1"/>
        <edge from-layer="8" from-port="2" to-layer="11" to-port="0"/>
        <edge from-layer="12" from-port="0" to-layer="11" to-port="1"/>
        <edge from-layer="11" from-port="2" to-layer="10" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer name="ExpandDims" id="2" type="PriorBoxClustered" precision="FP32">
            <data clip="0" step_h="16.000000" step_w="16.000000" flip="1" height="44,10,30,19,94,32,61,53,17" offset="0.500000" step="16.000000" variance="0.1,0.1,0.2,0.2" width="86,13,57,39,68,34,142,50,23"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 50, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 2;
                buffer[1] = 4;
                buffer[2] = 1;
                buffer[3] = 0;
            });
}

TEST_F(NGraphReaderTests, ReadPriorBoxNetwork) {
    std::string model = R"V0G0N(
<net name="PriorBoxNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,768,30,30"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,512,512"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ch_concat_mixed_7_chconcat_anchors/0_port" type="ShapeOf" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="1344813449_const" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="13458134591_const" type="Const" version="opset1">
            <data offset="8" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="1345813459_const" type="Const" version="opset1">
            <data offset="16" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="ch_concat_mixed_7_chconcat_anchors/ss_0_port" type="StridedSlice" version="opset1">
            <data begin_mask="0" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="ch_concat_mixed_7_chconcat_anchors/1_port" type="ShapeOf" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="ch_concat_mixed_7_chconcat_anchors/ss_1_port" type="StridedSlice" version="opset1">
            <data begin_mask="0" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="prior" type="PriorBox" version="opset1">
            <data density="" fixed_ratio="" fixed_size="" aspect_ratio="2.0,0.5" clip="0" flip="0" img_h="0" img_size="0" img_w="0" max_size="" min_size="51.2,72.407552" offset="0.5" scale_all_sizes="0" step="17.066666666666666" step_h="0" step_w="0" variance="0.100000,0.100000,0.200000,0.200000"/>
            <input>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </output>
        </layer>
        <layer id="12" name="ExpandAxis" type="Const" version="opset1">
            <data offset="24" size="8"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="11" name="ExpandDims" precision="FP32" type="Unsqueeze" version="opset1">
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="9" from-port="1" to-layer="5" to-port="3"/>
        <edge from-layer="9" from-port="1" to-layer="7" to-port="3"/>
        <edge from-layer="5" from-port="4" to-layer="8" to-port="0"/>
        <edge from-layer="7" from-port="4" to-layer="8" to-port="1"/>
        <edge from-layer="8" from-port="2" to-layer="11" to-port="0"/>
        <edge from-layer="12" from-port="0" to-layer="11" to-port="1"/>
        <edge from-layer="11" from-port="2" to-layer="10" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer name="ExpandDims" id="2" type="PriorBox" precision="FP32">
            <data density="" fixed_ratio="" fixed_size="" aspect_ratio="2,0.5" clip="0" flip="0" img_h="0" img_size="0" img_w="0" max_size="" min_size="51.200001,72.407555" offset="0.500000" scale_all_sizes="0" step="17.066668" step_h="0" step_w="0" variance="0.1,0.1,0.2,0.2"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 40, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 2;
                buffer[1] = 4;
                buffer[2] = 1;
                buffer[3] = 0;
            });
}
