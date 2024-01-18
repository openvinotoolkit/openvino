// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/test_common.hpp"


#include <ie/ie_core.hpp>

using namespace InferenceEngine;

namespace HeteroTests {

class HeteroQueryNetworkTest : public ::testing::TestWithParam<std::string>  {
public:
    void RunTest(std::string& deviceName) {
        ASSERT_GT(deviceName.size(), 0);

        //this model is a subgraph of "ctpn" model from omz
        std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="input" type="Parameter" version="opset1">
            <data shape="1,37,370,2" element_type="F32" />
            <output>
                <port id="0" precision="FP32" names="input">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="338" name="rpn_cls_prob/Transpose7580/value758213165" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="339" name="rpn_cls_prob/Transpose7580" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                </port>
            </output>
        </layer>
        <layer id="340" name="rpn_cls_prob/Transpose/value756213066" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="32" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="341" name="rpn_cls_prob/Transpose" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="rpn_cls_prob:0">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="342" name="Shape_2" type="ShapeOf" version="opset3">
            <data output_type="i32" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="343" name="Shape_2/GatherNCHWtoNHWC_input_port_1/value778413036" type="Const" version="opset1">
            <data element_type="i32" shape="4" offset="64" size="16" />
            <output>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="344" name="Shape_2/GatherNCHWtoNHWC_input_port_2/value778613297" type="Const" version="opset1">
            <data element_type="i64" shape="" offset="80" size="8" />
            <output>
                <port id="0" precision="I64" />
            </output>
        </layer>
        <layer id="345" name="Shape_2/GatherNCHWtoNHWC" type="Gather" version="opset8">
            <data batch_dims="0" />
            <input>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I32">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64" />
            </input>
            <output>
                <port id="3" precision="I32" names="Shape_2:0">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="346" name="strided_slice_6/stack" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="88" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_6/stack:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="347" name="strided_slice_6/stack_1" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="96" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_6/stack_1:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="348" name="strided_slice_6/stack_2" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="104" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_6/stack_2:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="349" name="strided_slice_6" type="StridedSlice" version="opset1">
            <data begin_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="1" ellipsis_mask="0" />
            <input>
                <port id="0" precision="I32">
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
                <port id="4" precision="I32" names="strided_slice_6:0" />
            </output>
        </layer>
        <layer id="350" name="Reshape_2/shape/Unsqueeze_input_port_1/value" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="112" size="8" />
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="351" name="Reshape_2/shape/Unsqueeze" type="Unsqueeze" version="opset1">
            <input>
                <port id="0" precision="I32" />
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="352" name="strided_slice_7/stack" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="120" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_7/stack:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="353" name="strided_slice_7/stack_1" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="128" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_7/stack_1:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="354" name="strided_slice_7/stack_2" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="136" size="8" />
            <output>
                <port id="0" precision="I64" names="strided_slice_7/stack_2:0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="355" name="strided_slice_7" type="StridedSlice" version="opset1">
            <data begin_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="1" ellipsis_mask="0" />
            <input>
                <port id="0" precision="I32">
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
                <port id="4" precision="I32" names="strided_slice_7:0" />
            </output>
        </layer>
        <layer id="356" name="Reshape_2/shape/Unsqueeze531_input_port_1/value" type="Const" version="opset1">
            <data element_type="i64" shape="1" offset="144" size="8" />
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="357" name="Reshape_2/shape/Unsqueeze531" type="Unsqueeze" version="opset1">
            <input>
                <port id="0" precision="I32" />
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="358" name="Reshape_2/shape/Unsqueeze533" type="Const" version="opset1">
            <data element_type="i32" shape="1" offset="152" size="4" />
            <output>
                <port id="0" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="359" name="Reshape_2/shape/Unsqueeze535" type="Const" version="opset1">
            <data element_type="i32" shape="1" offset="156" size="4" />
            <output>
                <port id="0" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="360" name="Reshape_2/shape" type="Concat" version="opset1">
            <data axis="0" />
            <input>
                <port id="0" precision="I32">
                    <dim>1</dim>
                </port>
                <port id="1" precision="I32">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I32">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I32">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="361" name="Reshape_2/Cast_1" type="Convert" version="opset1">
            <data destination_type="i64" />
            <input>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64" names="Reshape_2/shape:0">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="362" name="Reshape_2" type="Reshape" version="opset1">
            <data special_zero="false" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>370</dim>
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="Reshape_2,Reshape_2:0">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>37</dim>
                    <dim>20</dim>
                </port>
            </output>
        </layer>
        <layer id="363" name="Reshape_2:0" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>37</dim>
                    <dim>37</dim>
                    <dim>20</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0"   from-port="0" to-layer="339" to-port="0" />
        <edge from-layer="338" from-port="0" to-layer="339" to-port="1" />
        <edge from-layer="339" from-port="2" to-layer="341" to-port="0" />
        <edge from-layer="339" from-port="2" to-layer="342" to-port="0" />
        <edge from-layer="340" from-port="0" to-layer="341" to-port="1" />
        <edge from-layer="341" from-port="2" to-layer="362" to-port="0" />
        <edge from-layer="342" from-port="1" to-layer="345" to-port="0" />
        <edge from-layer="343" from-port="0" to-layer="345" to-port="1" />
        <edge from-layer="344" from-port="0" to-layer="345" to-port="2" />
        <edge from-layer="345" from-port="3" to-layer="355" to-port="0" />
        <edge from-layer="345" from-port="3" to-layer="349" to-port="0" />
        <edge from-layer="346" from-port="0" to-layer="349" to-port="1" />
        <edge from-layer="347" from-port="0" to-layer="349" to-port="2" />
        <edge from-layer="348" from-port="0" to-layer="349" to-port="3" />
        <edge from-layer="349" from-port="4" to-layer="351" to-port="0" />
        <edge from-layer="350" from-port="0" to-layer="351" to-port="1" />
        <edge from-layer="351" from-port="2" to-layer="360" to-port="0" />
        <edge from-layer="352" from-port="0" to-layer="355" to-port="1" />
        <edge from-layer="353" from-port="0" to-layer="355" to-port="2" />
        <edge from-layer="354" from-port="0" to-layer="355" to-port="3" />
        <edge from-layer="355" from-port="4" to-layer="357" to-port="0" />
        <edge from-layer="356" from-port="0" to-layer="357" to-port="1" />
        <edge from-layer="357" from-port="2" to-layer="360" to-port="1" />
        <edge from-layer="358" from-port="0" to-layer="360" to-port="2" />
        <edge from-layer="359" from-port="0" to-layer="360" to-port="3" />
        <edge from-layer="360" from-port="4" to-layer="361" to-port="0" />
        <edge from-layer="361" from-port="1" to-layer="362" to-port="1" />
        <edge from-layer="362" from-port="2" to-layer="363" to-port="0" />
    </edges>
</net>
)V0G0N";
        InferenceEngine::Core ie;
        Blob::Ptr weights;

        weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {160}, InferenceEngine::Layout::C));
        weights->allocate();

        auto *dataI64 = weights->buffer().as<int64_t *>();
        auto *dataI32 = weights->buffer().as<int32_t *>();
        dataI64[0] = 0;
        dataI64[1] = 3;
        dataI64[2] = 1;
        dataI64[3] = 2;

        dataI64[4] = 0;
        dataI64[5] = 2;
        dataI64[6] = 3;
        dataI64[7] = 1;

        dataI32[16] = 0;
        dataI32[17] = 2;
        dataI32[18] = 3;
        dataI32[19] = 1;

        dataI64[10] = 0;

        dataI64[11] = 0;
        dataI64[12] = 1;
        dataI64[13] = 1;

        dataI64[14] = 0;

        dataI64[15] = 1;
        dataI64[16] = 2;
        dataI64[17] = 1;

        dataI64[18] = 0;

        dataI32[38] = 0xFFFFFFFF;
        dataI32[39] = 20;

        auto network = ie.ReadNetwork(model, weights);

        QueryNetworkResult result;
        OV_ASSERT_NO_THROW(result = ie.QueryNetwork(network, deviceName));
        ASSERT_EQ(27, result.supportedLayersMap.size());

        std::set<std::string> checkNames = {"input",
            "rpn_cls_prob/Transpose7580/value758213165",
            "rpn_cls_prob/Transpose7580",
            "rpn_cls_prob/Transpose/value756213066",
            "rpn_cls_prob/Transpose",
            "Shape_2",
            "Shape_2/GatherNCHWtoNHWC_input_port_1/value778413036",
            "Shape_2/GatherNCHWtoNHWC_input_port_2/value778613297",
            "Shape_2/GatherNCHWtoNHWC",
            "strided_slice_6/stack",
            "strided_slice_6/stack_1",
            "strided_slice_6/stack_2",
            "strided_slice_6",
            "Reshape_2/shape/Unsqueeze_input_port_1/value",
            "Reshape_2/shape/Unsqueeze",
            "strided_slice_7/stack",
            "strided_slice_7/stack_1",
            "strided_slice_7/stack_2",
            "strided_slice_7",
            "Reshape_2/shape/Unsqueeze531_input_port_1/value",
            "Reshape_2/shape/Unsqueeze531",
            "Reshape_2/shape/Unsqueeze533",
            "Reshape_2/shape/Unsqueeze535",
            "Reshape_2/shape",
            "Reshape_2/Cast_1",
            "Reshape_2",
            "Reshape_2:0"};

        for (auto&& name : checkNames)
            EXPECT_NE(result.supportedLayersMap.find(name), result.supportedLayersMap.end());
    }
};

} // namespace HeteroTests
