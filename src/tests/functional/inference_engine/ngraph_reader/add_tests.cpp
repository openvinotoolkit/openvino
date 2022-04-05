// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadAddBcastNetwork) {
    std::string model = R"V0G0N(
<net name="add_bcast_model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,4,5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Parameter" version="opset1">
			<data element_type="f32" shape="5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="EltwiseReshapeNormalization/Cast_151_const" type="Const" version="opset1">
			<data element_type="i64" offset="0" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="EltwiseReshapeNormalization" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="sum" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="sum/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="add_bcast_model" version="7">
	<layers>
		<layer id="0" name="x" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="EltwiseReshapeNormalization/Cast_161_const" type="Const">
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" precision="I64" size="12"/>
			</blobs>
		</layer>
		<layer id="3" name="EltwiseReshapeNormalization" type="Reshape">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="sum" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
	</edges>
</net>
)V0G0N";

    compareIRs(model, modelV7, 24, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 1;
        data[2] = 5;
    });
}

TEST_F(NGraphReaderTests, ReadAddNetwork) {
    std::string model = R"V0G0N(
<net name="add_model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,4,5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,4,5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="sum" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sum/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
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
<net name="add_model" version="7">
	<layers>
		<layer id="0" name="x" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="sum" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
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

    compareIRs(model, modelV7, 0);
}
