// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "ngraph_reader_tests.hpp"

// since EinsumDecomposition is applied, disable these two tests
// until ngraph_reader_test checks only correctness of IR reading
TEST_F(NGraphReaderTests, DISABLED_ReadEinsumNetwork) {
  std::string model = R"V0G0N(
<net name="saved_model" version="10">
    <layers>
        <layer id="0" name="input_a" type="Parameter" version="opset1">
            <data shape="2,3,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_b" type="Parameter" version="opset1">
            <data shape="5,3,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>5</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="einsum" type="Einsum" version="opset7">
            <data equation="abc,dbc-&gt;ad"/>
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>5</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="gelu/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>2</dim>
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
<net name="saved_model" version="7">
    <layers>
        <layer id="0" name="input_a" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_b" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>5</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="einsum" type="Einsum" version="opset7">
            <data equation="abc,dbc-&gt;ad"/>
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>5</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
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
  compareIRs(model, modelV7);
}

TEST_F(NGraphReaderTests, DISABLED_ReadEinsumNetwork2) {
  std::string model = R"V0G0N(
<net name="saved_model" version="10">
    <layers>
        <layer id="0" name="input_a" type="Parameter" version="opset1">
            <data shape="2,3,4,5" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_b" type="Parameter" version="opset1">
            <data shape="4,5,6" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="input_c" type="Parameter" version="opset1">
            <data shape="7,4,5" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="einsum" type="Einsum" version="opset7">
            <data equation="abcd,cde,fcd-&gt;abe"/>
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>6</dim>
                </port>
                <port id="2">
                    <dim>7</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="gelu/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>6</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";
  std::string modelV7 = R"V0G0N(
<net name="saved_model" version="7">
    <layers>
        <layer id="0" name="input_a" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_b" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="input_c" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>7</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="einsum" type="Einsum" version="opset7">
            <data equation="abcd,cde,fcd-&gt;abe"/>
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>6</dim>
                </port>
                <port id="2">
                    <dim>7</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";
  compareIRs(model, modelV7);
}
