// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadProposalNetwork) {
    std::string model_v10 = R"V0G0N(
<net name="ProposalNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,12,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,24,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Const" version="opset1">
            <data offset="0" size="24"/>
            <output>
                <port id="0" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="3" version="opset1">
            <data feat_stride="16" base_size="16" min_size="16" ratio="2.669000" scale="4.000000,6.000000,9.000000,16.000000,24.000000,32.000000" pre_nms_topn="6000" post_nms_topn="200" nms_thresh="0.600000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="3">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="0"/>
    </edges>
    </net>
    )V0G0N";
    std::string model_v6  = R"V0G0N(
<net name="ProposalNet" version="6" batch="1">
    <layers>
        <layer name="in3" type="Const" precision="I64" id="4">
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24" />
            </blobs>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="5">
            <data base_size="16" box_coordinate_scale="1" box_size_scale="1" clip_after_nms="0" clip_before_nms="1" feat_stride="16" framework="" min_size="16" nms_thresh="0.600000023841858" normalize="0" post_nms_topn="200" pre_nms_topn="6000" ratio="2.668999910354614" scale="4,6,9,16,24,32" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1" />
        <edge from-layer="2" from-port="0" to-layer="5" to-port="0" />
        <edge from-layer="4" from-port="2" to-layer="5" to-port="2" />
    </edges>
    <statistics />
</net>
)V0G0N";

    compareIRs(model_v10, model_v6, 24);
}

TEST_F(NGraphReaderTests, ReadProposalNetwork_2) {
    std::string model_v10 = R"V0G0N(
<net name="ProposalNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,12,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,24,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Const" version="opset1">
            <data offset="0" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="3" version="opset1">
            <data feat_stride="16" base_size="16" min_size="16" ratio="2.669000" scale="4.000000,6.000000,9.000000,16.000000,24.000000,32.000000" pre_nms_topn="6000" post_nms_topn="200" nms_thresh="0.600000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="3">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="0"/>
    </edges>
    </net>
    )V0G0N";
    std::string model_v6  = R"V0G0N(
<net name="ProposalNet" version="6" batch="1">
    <layers>
        <layer name="in3" type="Const" precision="I64" id="4">
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32" />
            </blobs>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="5">
            <data base_size="16" box_coordinate_scale="1" box_size_scale="1" clip_after_nms="0" clip_before_nms="1" feat_stride="16" framework="" min_size="16" nms_thresh="0.600000023841858" normalize="0" post_nms_topn="200" pre_nms_topn="6000" ratio="2.668999910354614" scale="4,6,9,16,24,32" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1" />
        <edge from-layer="2" from-port="0" to-layer="5" to-port="0" />
        <edge from-layer="4" from-port="2" to-layer="5" to-port="2" />
    </edges>
    <statistics />
</net>
)V0G0N";

    compareIRs(model_v10, model_v6, 32);
}
