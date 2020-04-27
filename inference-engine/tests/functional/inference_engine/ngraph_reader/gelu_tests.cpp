// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"

TEST_F(NGraphReaderTests, ReadGeluNetwork) {
    CommonTestUtils::IRBuilder_v10 ir_builder_v10("Network");

    auto input_layer = ir_builder_v10
            .AddLayer("in1", "Parameter", {{"shape", "1,128"},
                                           {"element_type", "f32"}}).AddOutPort(Precision::ePrecision::FP32, {1, 128})
            .getLayer();

    auto gelu_layer = ir_builder_v10
            .AddLayer("activation", "Gelu", {}, "opset2")
            .AddInPort(Precision::ePrecision::FP32, {1, 128})
            .AddOutPort(Precision::ePrecision::FP32, {1, 128})
            .getLayer();

    auto result_layer = ir_builder_v10
            .AddLayer("output", "Result")
            .AddInPort(Precision::ePrecision::FP32, {1, 128})
            .getLayer();

    input_layer.out(0).connect(gelu_layer.in(0));
    gelu_layer.out(0).connect(result_layer.in(0));

    // f(x) = 0.5 * x * (1.0 + erf( x / sqrt(2.0) )
    std::string model_v7 = R"V0G0N(
<?xml version="1.0"?>
<net name="Network" version="6" batch="1">
	<layers>
		<layer name="in1" type="Input" precision="FP32" id="0">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer name="Multiply_304" type="Power" precision="FP32" id="1">
			<data power="1" scale="0.5" shift="0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer name="Divide_307" type="Power" precision="FP32" id="2">
			<data power="1" scale="0.707107" shift="0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer name="Erf_308" type="Erf" precision="FP32" id="3">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer name="Add_310" type="Power" precision="FP32" id="4">
			<data power="1" scale="1" shift="1" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer name="activation" type="Eltwise" precision="FP32" id="5">
			<data operation="prod" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
		<edge from-layer="1" from-port="1" to-layer="5" to-port="0" />
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
		<edge from-layer="4" from-port="1" to-layer="5" to-port="1" />
	</edges>
	<statistics />
</net>
    )V0G0N";

    std::string model_v10 = ir_builder_v10.serialize();

    compareIRs(model_v10, model_v7, 0);
}
