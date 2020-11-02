// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadReduceMinNetwork) {
    std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,2,2"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="reduced/Cast_175_const" type="Const" version="opset1">
			<data element_type="i64" offset="0" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="reduced" type="ReduceMin" version="opset1">
			<data keep_dims="True"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="reduced/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="data" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="reduced/Cast_184_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" precision="I64" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="reduced" type="ReduceMin" version="opset1">
			<data keep_dims="True"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 100, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 0;
                buffer[1] = 1;
                buffer[2] = 2;
             });
}
