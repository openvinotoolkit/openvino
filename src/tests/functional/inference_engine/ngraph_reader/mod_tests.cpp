// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>


TEST_F(NGraphReaderTests, ReadModNetwork) {
    std::string modelV10 = R"V0G0N(
<net name="Mod" version="10">
	<layers>
		<layer id="0" name="input_a" type="Parameter" version="opset1">
			<data shape="256,56" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="input_b" type="Parameter" version="opset1">
			<data shape="256,56" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="Mod" version="opset1">
			<input>
				<port id="0">
					<dim>256</dim>
					<dim>56</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>256</dim>
					<dim>56</dim>
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
    Core ie;
    Blob::Ptr weights;
    std::shared_ptr<ngraph::Function> f_ref{nullptr};

    auto data_A = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    auto data_B = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    auto mod = std::make_shared<ngraph::opset1::Mod>(data_A, data_B);
    f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mod}, ngraph::ParameterVector{data_A, data_B});

    auto network = ie.ReadNetwork(modelV10, weights);
    auto f = network.getFunction();

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
