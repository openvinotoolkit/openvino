// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ConvertStridedSliceToCrop) {
    std::string model_version10 = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0">
					<dim>300</dim>
					<dim>90</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
            </output>
        </layer>
		<layer id="1" name="const1" precision="I64" type="Const">
			<data offset="0" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="const1" precision="I64" type="Const">
			<data offset="32" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="Stride" precision="I64" type="Const">
			<data offset="64" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Crop_" precision="FP32" type="StridedSlice">
			<data begin_mask="0,1,0,0" ellipsis_mask="0,0,0,0" end_mask="0,1,0,0" new_axis_mask="0,0,0,0" shrink_axis_mask="0,0,0,0"/>
			<input>
				<port id="0">
					<dim>300</dim>
					<dim>90</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
				<port id="2">
					<dim>4</dim>
				</port>
				<port id="3">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>300</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="5" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
	                <dim>300</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string model_version6 = R"V0G0N(
<net name="Reshape" version="6" batch="300">
	<layers>
		<layer name="data" type="Input" precision="UNSPECIFIED" id="0">
			<output>
				<port id="0">
					<dim>300</dim>
					<dim>90</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer name="Crop_" type="Crop" precision="UNSPECIFIED" id="1">
			<data axis="0,1,2,3" dim="300,1,1,4" offset="0,1,0,0" />
			<input>
				<port id="0">
					<dim>300</dim>
					<dim>90</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>300</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
	</edges>
	<statistics />
</net>
)V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {96}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();

    // According to begin (0,1,0,0) and end masks (0,1,0,0)
    // and input and result shapes (300, 90, 1, 4) -> (300, 1, 1, 4)
    data[1] = 1;
    data[5] = 2;

    // Set "1" into each stride to apply "StrideSliceToCrop" transformation
    for (int stride_node_idx = 8; stride_node_idx < 12; ++stride_node_idx){
        data[stride_node_idx] = 1;
    }

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model_version10, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model_version6.data(), model_version6.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}
