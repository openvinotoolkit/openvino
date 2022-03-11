// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadCTCGreedyDecoderNetwork) {
    std::string model = R"V0G0N(
<net name="ctcgreedydecoder" version="10">
	<layers>
		<layer id="0" name="CTCGreedyDecoder/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="20,8,128" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Fill/Broadcast/Output_0/Data__const" type="Const" version="opset1">
			<data offset="0" size="640" shape="20,8" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="CTCGreedyDecoder" type="CTCGreedyDecoder" version="opset1">
			<data ctc_merge_repeated="1"/>
			<input>
				<port id="0">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="CTCGreedyDecoder/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
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
<net name="ctcgreedydecoder" version="7">
	<layers>
		<layer id="0" name="CTCGreedyDecoder/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Fill/Broadcast/Output_0/Data__const" type="Const" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="640" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="2" name="CTCGreedyDecoder" type="CTCGreedyDecoder" version="opset1">
			<data ctc_merge_repeated="1"/>
			<input>
				<port id="0">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
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
    compareIRs(model, modelV7, 640);
}


#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadCTCGreedyDecoderNoMergeNetwork) {
    std::string model = R"V0G0N(
<net name="ctcgreedydecoder" version="10">
	<layers>
		<layer id="0" name="CTCGreedyDecoder/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="20,8,128" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Fill/Broadcast/Output_0/Data__const" type="Const" version="opset1">
			<data offset="0" size="640" shape="20,8" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="CTCGreedyDecoder" type="CTCGreedyDecoder" version="opset1">
			<data ctc_merge_repeated="0"/>
			<input>
				<port id="0">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="CTCGreedyDecoder/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
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
<net name="ctcgreedydecoder" version="7">
	<layers>
		<layer id="0" name="CTCGreedyDecoder/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Fill/Broadcast/Output_0/Data__const" type="Const" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="640" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="2" name="CTCGreedyDecoder" type="CTCGreedyDecoder" version="opset1">
			<data ctc_merge_repeated="0"/>
			<input>
				<port id="0">
					<dim>20</dim>
					<dim>8</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>8</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
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
    compareIRs(model, modelV7, 640);
}
