// Copyright (C) 2018-2022 Intel Corporation
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
		<layer id="2" name="365" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>104</dim>
					<dim>32</dim>
					<dim>13</dim>
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
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7);
}
