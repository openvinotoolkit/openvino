// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadUnsqueeze) {
    std::string model_version10 = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="const1" precision="I64" type="Const">
			<data offset="0" size="8"/>
			<output>
				<port id="0">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="ExpandDims" precision="FP32" type="Unsqueeze">
			<input>
				<port id="0">
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
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
    std::string model_version6 = R"V0G0N(
<net name="Reshape" version="6" batch="1">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<output>
				<port id="0">
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer name="const1" type="Const" precision="I64" id="1">
			<output>
				<port id="0">
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="8" />
			</blobs>
		</layer>
		<layer name="ExpandDims" type="Unsqueeze" precision="FP32" id="2">
			<input>
				<port id="0">
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>32</dim>
					<dim>64</dim>
					<dim>60</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
	</edges>
	<statistics />
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 3;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph = reader.read(model_version10, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model_version6.data(), model_version6.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}