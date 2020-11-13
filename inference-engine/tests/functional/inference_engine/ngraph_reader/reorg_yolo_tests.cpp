// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadReorgYoloNetwork) {
    std::string model = R"V0G0N(
<net name="frozen_graph" version="10">
	<layers>
		<layer id="0" name="yolov2/yolov2_feature/lower_features/downsample/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="1,26,64,26" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>26</dim>
					<dim>64</dim>
					<dim>26</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolov2/yolov2_feature/lower_features/downsample" type="ReorgYolo" version="opset2">
			<data stride="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>26</dim>
					<dim>64</dim>
					<dim>26</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>104</dim>
					<dim>32</dim>
					<dim>13</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="yolov2/yolov2_feature/lower_features/downsample/YoloRegion" type="RegionYolo" version="opset1">
			<data coords="4" classes="20" mask="0" num="5" axis="1" end_axis="3" do_softmax="1" anchors="1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>104</dim>
					<dim>32</dim>
					<dim>13</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>43264</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="365" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>43264</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="frozen_graph" version="7">
	<layers>
		<layer id="0" name="yolov2/yolov2_feature/lower_features/downsample/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>26</dim>
					<dim>64</dim>
					<dim>26</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolov2/yolov2_feature/lower_features/downsample" type="ReorgYolo" version="opset2">
			<data stride="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>26</dim>
					<dim>64</dim>
					<dim>26</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>104</dim>
					<dim>32</dim>
					<dim>13</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="yolov2/yolov2_feature/lower_features/downsample/YoloRegion" type="RegionYolo" version="opset1">
			<data coords="4" mask="0" classes="20" num="5" axis="1" end_axis="3" do_softmax="1" anchors="1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>104</dim>
					<dim>32</dim>
					<dim>13</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>43264</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7);
}
