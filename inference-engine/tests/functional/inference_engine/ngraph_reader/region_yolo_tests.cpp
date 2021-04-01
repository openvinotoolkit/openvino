// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadRegionYoloNetwork) {
    std::string model = R"V0G0N(
<net name="frozen_graph" version="10">
	<layers>
		<layer id="0" name="yolo_out_postprocess/placeholder_port_0" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,125,13,13"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>125</dim>
					<dim>13</dim>
					<dim>13</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolo_out_postprocess/YoloRegion" type="RegionYolo" version="opset1">
			<data coords="4" mask="0" classes="20" num="5" axis="1" end_axis="3" do_softmax="1" anchors="1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>125</dim>
					<dim>13</dim>
					<dim>13</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>21125</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="364" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>21125</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="frozen_graph" version="7">
	<layers>
		<layer id="0" name="yolo_out_postprocess/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>125</dim>
					<dim>13</dim>
					<dim>13</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolo_out_postprocess/YoloRegion" type="RegionYolo" version="opset1">
			<data coords="4" classes="20" num="5" axis="1" end_axis="3" do_softmax="1" mask="0" anchors="1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>125</dim>
					<dim>13</dim>
					<dim>13</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>21125</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 4);
}


TEST_F(NGraphReaderTests, ReadRegionYoloMaskNetwork) {
    std::string model = R"V0G0N(
<net name="frozen_graph" version="10">
	<layers>
		<layer id="0" name="yolo_out_postprocess/placeholder_port_0" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,255,26,26"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolo_out_postprocess/YoloRegion" type="RegionYolo" version="opset1">
			<data anchors="10,14,23,27,37,58,81,82,135,169,344,319" axis="1" classes="80" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="6"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="364" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="frozen_graph" version="7">
	<layers>
		<layer id="0" name="yolo_out_postprocess/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="yolo_out_postprocess/YoloRegion" type="RegionYolo" version="opset1">
		    <data anchors="10,14,23,27,37,58,81,82,135,169,344,319" axis="1" classes="80" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="6"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>255</dim>
					<dim>26</dim>
					<dim>26</dim>>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 4);
}
