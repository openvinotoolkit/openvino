// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadMatMulNetwork1) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="2048,1000" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // 'fc' layer biases are fake and added due to IE limitation for Fully Connected layer
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
            <biases offset="8192000" size="1000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8193000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork2) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="1000,2048" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <data transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // 'fc' layer biases are fake and added due to IE limitation for Fully Connected layer
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
            <biases offset="8192000" size="1000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8193000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork3) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="2048,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="1000,2048" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <data transpose_a="True" transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // 'fc' layer biases are fake and added due to IE limitation for FUlly Connected layer
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="fc/transpose_a" precision="FP32" type="Permute">
            <data order="1,0" originalLayersNames="fc"/>
            <input>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
            <biases offset="8192000" size="1000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8193000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork4) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data1" type="Parameter" version="opset1">
            <data element_type="f32" shape="2048,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul" version="opset1">
            <data transpose_a="True" transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" precision="FP32" type="Gemm">
            <data transpose_a="true" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1000</dim>
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
    compareIRs(model, modelV5, 8192000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork5) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data1" type="Parameter" version="opset1">
            <data element_type="f32" shape="2,3,2"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" type="Parameter" version="opset1">
            <data element_type="f32" shape="3,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Constant" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="fc/reshape" precision="FP32" type="Reshape">
            <data originalLayersNames="fc"/>
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="Gemm">
            <data transpose_a="false" transpose_b="false" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 48);
}

TEST_F(NGraphReaderTests, ReadMatMul1DNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="2048,1000" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // 'fc' layer biases are fake and added due to IE limitation for Fully Connected layer
    std::string modelV5 = R"V0G0N(
<?xml version="1.0"?>
<net name="Network" version="6" batch="1">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<data originalLayersNames="data" />
			<output>
				<port id="0" precision="FP32">
					<dim>2048</dim>
				</port>
			</output>
		</layer>
		<layer name="Constant_735" type="Const" precision="I64" id="1">
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="16" precision="I64" />
			</blobs>
		</layer>
		<layer name="fc/Reshape" type="Reshape" precision="FP32" id="2">
			<data dim="" originalLayersNames="fc" />
			<input>
				<port id="0">
					<dim>2048</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2048</dim>
				</port>
			</output>
		</layer>
		<layer name="FullyConnected_737" type="FullyConnected" precision="FP32" id="3">
			<data originalLayersNames="fc" out-size="1000" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2048</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
			<blobs>
				<biases offset="16" size="4000" precision="FP32" />
				<weights offset="4016" size="8192000" precision="FP32" />
			</blobs>
		</layer>
		<layer name="Constant_738" type="Const" precision="I64" id="4">
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="8196016" size="8" precision="I64" />
			</blobs>
		</layer>
		<layer name="fc" type="Reshape" precision="FP32" id="5">
			<data dim="" originalLayersNames="fc" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1000</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
		<edge from-layer="3" from-port="1" to-layer="5" to-port="0" />
		<edge from-layer="4" from-port="0" to-layer="5" to-port="1" />
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8293000);
}
