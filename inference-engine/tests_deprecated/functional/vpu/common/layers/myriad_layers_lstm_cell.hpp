// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

using namespace InferenceEngine;

const int gate_map[] = {0, 1, 3, 2};
const size_t ngates = 4;
#define ERROR_BOUND (.01f)

std::string tensorIteratorModel_1 = R"V0G0N(
<net batch="1" name="ctpn" version="4">
  <layers>
    <layer id="0" name="RNNInput_Hidden" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="RNNInput_CellState" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="RNNInput" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
      </output>
    </layer>
    <layer id="38" name="RNNOutput" precision="FP16" type="TensorIterator">
      <input>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
        <port id="1">
          <dim>4</dim>
          <dim>128</dim>
        </port>
        <port id="2">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </input>
      <output>
        <port id="3">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
      </output>
      <port_map>
        <input axis="1" external_port_id="0" internal_layer_id="0" internal_port_id="0"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        <input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
        <output axis="1" external_port_id="3" internal_layer_id="2" internal_port_id="1"/>
      </port_map>
      <back_edges>
        <edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
      </back_edges>
      <body>
        <layers>
          <layer id="0" name="lstm_o/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP16" type="Reshape">
            <data dim="-1,512"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>1</dim>
                <dim>512</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>512</dim>
              </port>
            </output>
          </layer>
          <layer id="1" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell" precision="FP16" type="LSTMCell">
            <data hidden_size="128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>512</dim>
              </port>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="2">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="5">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="6">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
            <blobs>
              <weights offset="0" size="655360"/>
              <biases offset="655360" size="1024"/>
            </blobs>
          </layer>
          <layer id="2" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
        </layers>
        <edges>
          <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
          <edge from-layer="1" from-port="5" to-layer="2" to-port="0"/>
        </edges>
      </body>
    </layer>
  </layers>
  <edges>
    <edge from-layer="2" from-port="0" to-layer="38" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="38" to-port="1"/>
    <edge from-layer="1" from-port="0" to-layer="38" to-port="2"/>
  </edges>
</net>
)V0G0N";

std::string tensorIteratorModel_2 = R"V0G0N(
<net batch="1" name="ctpn" version="4">
  <layers>
    <layer id="0" name="RNNInput_Hidden" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="RNNInput_CellState" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="RNNInput" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
      </output>
    </layer>
    <layer id="38" name="RNNOutput" precision="FP16" type="TensorIterator">
      <input>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
        <port id="1">
          <dim>4</dim>
          <dim>128</dim>
        </port>
        <port id="2">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </input>
      <output>
        <port id="3">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
        <port id="4">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
      <port_map>
        <input external_port_id="0" internal_layer_id="1" internal_port_id="0"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        <input axis="0" external_port_id="2" internal_layer_id="1" internal_port_id="2"/>

        <output external_port_id="3" internal_layer_id="2" internal_port_id="1"/>
        <output axis="0" external_port_id="4" internal_layer_id="3" internal_port_id="1"/>
      </port_map>
      <back_edges>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="6" to-layer="3" to-port="0"/>
      </back_edges>
      <body>
        <layers>
          <layer id="0" name="lstm_o/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP16" type="Reshape">
            <data dim="-1,512"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>1</dim>
                <dim>512</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>512</dim>
              </port>
            </output>
          </layer>
          <layer id="1" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell" precision="FP16" type="LSTMCell">
            <data hidden_size="128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>512</dim>
              </port>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="2">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="5">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="6">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
            <blobs>
              <weights offset="0" size="655360"/>
              <biases offset="655360" size="1024"/>
            </blobs>
          </layer>
          <layer id="2" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,128"/>
			<input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
          <layer id="3" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_1/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,128"/>
			<input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
        </layers>
        <edges>
			
			<edge from-layer="2" from-port="1" to-layer="1" to-port="1"/>
			<edge from-layer="3" from-port="1" to-layer="1" to-port="2"/>
        </edges>
      </body>
    </layer>
  </layers>
  <edges>
    <edge from-layer="2" from-port="0" to-layer="38" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="38" to-port="1"/>
    <edge from-layer="1" from-port="0" to-layer="38" to-port="2"/>
  </edges>
</net>
)V0G0N";

std::string tensorIteratorModel_4 = R"V0G0N(
<net batch="1" name="ctpn" version="4">
  <layers>
    <layer id="0" name="RNNInput_Hidden" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="RNNInput_CellState" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="RNNInput" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
      </output>
    </layer>
    <layer id="38" name="RNNOutput" precision="FP16" type="TensorIterator">
      <input>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
        <port id="1">
          <dim>4</dim>
          <dim>128</dim>
        </port>
        <port id="2">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </input>
      <output>
        <port id="3">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
		<port id="4">
          <dim>4</dim>
		  <dim>2</dim>
          <dim>128</dim>
        </port>
      </output>
      <port_map>
        <input axis="1" external_port_id="0" internal_layer_id="0" internal_port_id="0"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        <input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
        <output axis="1" external_port_id="3" internal_layer_id="2" internal_port_id="1"/>
		<output axis="1" external_port_id="4" internal_layer_id="3" internal_port_id="1"/>
      </port_map>
      <back_edges>
        <edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
      </back_edges>
      <body>
        <layers>
          <layer id="0" name="lstm_o/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP16" type="Reshape">
            <data dim="-1,512"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>1</dim>
                <dim>512</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>512</dim>
              </port>
            </output>
          </layer>
          <layer id="1" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell" precision="FP16" type="LSTMCell">
            <data hidden_size="128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>512</dim>
              </port>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="2">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="5">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="6">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
            <blobs>
              <weights offset="0" size="655360"/>
              <biases offset="655360" size="1024"/>
            </blobs>
          </layer>
          <layer id="2" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
		  <layer id="3" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_1/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
				<dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
        </layers>
        <edges>
          <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
          <edge from-layer="1" from-port="5" to-layer="2" to-port="0"/>
		  <edge from-layer="1" from-port="6" to-layer="3" to-port="0"/>
        </edges>
      </body>
    </layer>
  </layers>
  <edges>
    <edge from-layer="2" from-port="0" to-layer="38" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="38" to-port="1"/>
    <edge from-layer="1" from-port="0" to-layer="38" to-port="2"/>
  </edges>
</net>
)V0G0N";

std::string tensorIteratorModel_31 = R"V0G0N(
<net batch="1" name="ctpn" version="4">
  <layers>
    <layer id="0" name="RNNInput_Hidden" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="RNNInput_CellState" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="RNNInput" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
      </output>
    </layer>
    <layer id="38" name="RNNOutput" precision="FP16" type="TensorIterator">
      <input>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
        <port id="1">
          <dim>4</dim>
          <dim>128</dim>
        </port>
        <port id="2">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </input>
      <output>
        <port id="3">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
		<port id="4">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
		<port id="5">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
      </output>
      <port_map>
        <input axis="1" external_port_id="0" internal_layer_id="0" internal_port_id="0"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        <input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
        <output axis="1" external_port_id="3" internal_layer_id="2" internal_port_id="1"/>
		<output axis="1" external_port_id="4" internal_layer_id="3" internal_port_id="1"/>
		<output axis="1" external_port_id="5" internal_layer_id="4" internal_port_id="1"/>
      </port_map>
      <back_edges>
        <edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
      </back_edges>
      <body>
        <layers>
          <layer id="0" name="lstm_o/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP16" type="Reshape">
            <data dim="-1,512"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>1</dim>
                <dim>512</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>512</dim>
              </port>
            </output>
          </layer>
          <layer id="1" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell" precision="FP16" type="LSTMCell">
            <data hidden_size="128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>512</dim>
              </port>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="2">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="5">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="6">
                <dim>4</dim>
                <dim>128</dim>
              </port>
			  <port id="7">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
            <blobs>
              <weights offset="0" size="655360"/>
              <biases offset="655360" size="1024"/>
            </blobs>
          </layer>
          <layer id="2" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
		  <layer id="3" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_1/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
		  <layer id="4" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_2/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,1,128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>1</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
        </layers>
        <edges>
          <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
          <edge from-layer="1" from-port="5" to-layer="2" to-port="0"/>
		  <edge from-layer="1" from-port="6" to-layer="3" to-port="0"/>
		  <edge from-layer="1" from-port="7" to-layer="4" to-port="0"/>
        </edges>
      </body>
    </layer>
  </layers>
  <edges>
    <edge from-layer="2" from-port="0" to-layer="38" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="38" to-port="1"/>
    <edge from-layer="1" from-port="0" to-layer="38" to-port="2"/>
  </edges>
</net>
)V0G0N";

std::string tensorIteratorModel_3 = R"V0G0N(
<net batch="1" name="ctpn" version="4">
  <layers>
    <layer id="0" name="RNNInput_Hidden" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="RNNInput_CellState" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="RNNInput" precision="FP16" type="Input">
      <output>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
      </output>
    </layer>
    <layer id="38" name="RNNOutput" precision="FP16" type="TensorIterator">
      <input>
        <port id="0">
          <dim>4</dim>
          <dim>2</dim>
          <dim>512</dim>
        </port>
        <port id="1">
          <dim>4</dim>
          <dim>128</dim>
        </port>
        <port id="2">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </input>
      <output>
        <port id="3">
          <dim>4</dim>
          <dim>2</dim>
          <dim>128</dim>
        </port>
        <port id="4">
          <dim>4</dim>
          <dim>128</dim>
        </port>
      </output>
      <port_map>
        <input axis="0" external_port_id="0" internal_layer_id="1" internal_port_id="0"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        <input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>

        <output external_port_id="3" internal_layer_id="2" internal_port_id="1"/>
        <output axis="0" external_port_id="4" internal_layer_id="3" internal_port_id="1"/>
      </port_map>
      <back_edges>
        <edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
      </back_edges>
      <body>
        <layers>
          <layer id="0" name="lstm_o/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP16" type="Reshape">
            <data dim="-1,512"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>1</dim>
                <dim>512</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>512</dim>
              </port>
            </output>
          </layer>
          <layer id="1" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell" precision="FP16" type="LSTMCell">
            <data hidden_size="128"/>
            <input>
              <port id="0">
                <dim>4</dim>
                <dim>512</dim>
              </port>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="2">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="5">
                <dim>4</dim>
                <dim>128</dim>
              </port>
              <port id="6">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
            <blobs>
              <weights offset="0" size="655360"/>
              <biases offset="655360" size="1024"/>
            </blobs>
          </layer>
          <layer id="2" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,128"/>
			      <input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
          <layer id="3" name="lstm_o/bidirectional_rnn/fw/fw/while/fw/lstm_cell/concat/LSTMCell/Output_1/Data_/OutputUnsqueeze" precision="FP16" type="Reshape">
            <data dim="-1,128"/>
		      	<input>
              <port id="0">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </input>
            <output>
              <port id="1">
                <dim>4</dim>
                <dim>128</dim>
              </port>
            </output>
          </layer>
        </layers>
        <edges>
          <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
          <edge from-layer="1" from-port="5" to-layer="2" to-port="0"/>
          <edge from-layer="1" from-port="6" to-layer="3" to-port="0"/>
        </edges>
      </body>
    </layer>
  </layers>
  <edges>
    <edge from-layer="2" from-port="0" to-layer="38" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="38" to-port="1"/>
    <edge from-layer="1" from-port="0" to-layer="38" to-port="2"/>
  </edges>
</net>
)V0G0N";

struct  lstmcell_test_params {
    int input_size;
    int state_size;

    int output_num;
    friend std::ostream& operator<<(std::ostream& os, lstmcell_test_params const& tst)
    {
        return os << " input size = " << tst.input_size
                  << ", state size = " << tst.state_size;
    };
};
typedef myriadLayerTestBaseWithParam<lstmcell_test_params> myriadLayersTestsLSTMCell_nightly;

#define f32Tof16 PrecisionUtils::f32tof16
#define f16Tof32 PrecisionUtils::f16tof32
static ie_fp16& at(ie_fp16 *a, int i, int j, int k, int stride0, int stride1)
{
    return *(i * stride1 + j * stride0 + k + a);
}
static ie_fp16& at(ie_fp16 *a, int i, int j, int stride)
{
    return *(i * stride + j + a);
}
// float a[m][k], float b[k][n], float c[m][n];
// c = a * b;
static void gemm(int m, int n, int k,
                 ie_fp16 * a, int stride_a,
                 ie_fp16 * b, int stride_b,
                 ie_fp16 * c, int stride_c,
                 ie_fp16 beta) {
    for (int im = 0; im < m; im++) {
        for (int in = 0; in < n; in++) {
            // if beta == 0 the initialize pc by 0. Multiplication of
            // uninitialized value even by zero can lead to nan
            ie_fp16 c_elem = (beta == (ie_fp16)0.) ? (ie_fp16)0. : f32Tof16(f16Tof32(at(c, im, in, stride_c)) * f16Tof32(beta));
            for (int ik = 0; ik < k; ik++) {
                ie_fp16 a_elem = at(a, im, ik, stride_a);
                ie_fp16 b_elem = at(b, ik, in, stride_b);
                c_elem = f32Tof16(f16Tof32(a_elem) * f16Tof32(b_elem) + f16Tof32(c_elem));
            }
            at(c, im, in, stride_c) = c_elem;
        }
    }
}

static float logistic(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static void lstm_activation(int dic, int n_gates, int batch, ie_fp16 * a) {
    for (int ib = 0; ib < batch; ib++) {
        for (int ig = 0; ig < 3; ig++) {
            for (int ih = 0; ih < dic; ih++) {
                *(a + ih + ig * dic + ib * dic * n_gates) = f32Tof16(logistic(f16Tof32(*(a + ih + ig * dic + ib * dic * n_gates))));
            }
        }
        int ig = 3;
        for (int j = 0; j < dic; j++) {
            *(a + j + ig * dic + ib * dic * n_gates) = f32Tof16(tanhf(f16Tof32(*(a + j + ig * dic + ib * dic * n_gates))));
        }
    }
}

// src_layer[input_size]
// src_iter_h[state_size]
// src_iter_c[state_size]
// weights_layer[ngates * state_size][input_size]
// weights_iter_h[ngates * state_size][state_size]
// bias[ngates][state_size]
// h_dst[state_size]
// c_dst[state_size]
void lstm_cell(int input_size,
               int state_size,
               // weights
               ie_fp16* weights_layer,
               ie_fp16* weights_iter_h,
               ie_fp16* bias,
               // input
               ie_fp16* src_layer,
               ie_fp16* src_iter_h,
               ie_fp16* src_iter_c,

               int output_num,
               // output
               ie_fp16* h_dst,
               ie_fp16* c_dst,
               ie_fp16* l_h_dst,

               ie_fp16* gates
              )
{
    const int n_gates = 4;
    const int ohf = 0; const int ohi = 1; const int oho = 2; const int ohc = 3;

    int num_weights = state_size * (input_size + state_size);
    int num_bias = state_size;

    /* gates = src_layer * weights_layer */
    gemm(1, n_gates * state_size, input_size,
         src_layer,     input_size,
         weights_layer, n_gates * state_size,
         gates,         n_gates * state_size,
         f32Tof16(0.0f));

    /* gates += src_iter_h * weights_iter_h */
    gemm(1, n_gates * state_size, state_size,
         src_iter_h,     state_size,
         weights_iter_h, n_gates * state_size,
         gates,          n_gates * state_size,
         f32Tof16(1.0f));

    // add bias
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < n_gates; j++) {
            for (int k = 0; k < state_size; k++) {
                *(gates + i * n_gates * state_size + j * state_size + k) =
                f32Tof16(
                         f16Tof32(*(gates + i * n_gates * state_size + j * state_size + k))
                       + f16Tof32(*(bias + j * state_size + k))
                        );
            }
        }
    }
    // run the eltwise
    lstm_activation(state_size, n_gates, 1, gates);
    // compute C_t_l and H_t_l
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < state_size; j++) {
            float tmp = f16Tof32(at(gates, i, ohf, j, state_size, state_size * n_gates)) *
                        f16Tof32(at(src_iter_c, i, j, state_size))
                      + f16Tof32(at(gates, i, ohi, j, state_size, state_size * n_gates)) *
                        f16Tof32(at(gates, i, ohc, j, state_size, state_size * n_gates));
            at(c_dst, i, j, state_size) = f32Tof16(tmp);
            at(h_dst, i, j, state_size) = f32Tof16(f16Tof32(at(gates, i, oho, j, state_size, state_size * n_gates)) * tanhf(tmp));
            if (output_num == 3 && l_h_dst) {
              at(l_h_dst, i, j, state_size) = at(h_dst, i, j, state_size);
            }
        }
    }
}
/* psrc[m][n] -> pdst[n][m] */
static void matrix_copy_transpose(const ie_fp16 *psrc, ie_fp16 *pdst, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            pdst[j * m + i] = psrc[i * n + j];
        }
    }
}

/* psrc[m][n][k] -> pdst[k][m][n] */
static void matrix_copy_transpose_repack(const ie_fp16 *psrc, ie_fp16 *pdst, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                pdst[l * m * n + i * n + j] = psrc[i * m * n + j * n + l];
            }
        }
    }
}

TEST_P(myriadLayersTestsLSTMCell_nightly, LSTMCell) {
    auto param = GetParam();
    lstmcell_test_params test_params = param;

    size_t input_size = param.input_size;
    size_t state_size = param.state_size;

    size_t num_weights = ngates * state_size * (input_size + state_size);
    size_t num_bias = ngates * state_size;

    IN_OUT_desc dims_input;
    
    dims_input.resize(3);
    /* inputs */
    dims_input[0].resize(2);
    dims_input[0][0] = 1;
    dims_input[0][1] = input_size;
    dims_input[1].resize(2);
    dims_input[1][0] = 1;
    dims_input[1][1] = state_size;
    dims_input[2].resize(2);
    dims_input[2][0] = 1;
    dims_input[2][1] = state_size;

    IN_OUT_desc dims_output;

    if (param.output_num == 3) {
      dims_output.resize(3);
      /* outputs */
      dims_output[0].resize(2);
      dims_output[0][0] = 1;
      dims_output[0][1] = state_size;
      dims_output[1].resize(2);
      dims_output[1][0] = 1;
      dims_output[1][1] = state_size;
      dims_output[2].resize(2);
      dims_output[2][0] = 1;
      dims_output[2][1] = state_size;
    } else {
      dims_output.resize(2);
      /* outputs */
      dims_output[0].resize(2);
      dims_output[0][0] = 1;
      dims_output[0][1] = state_size;
      dims_output[1].resize(2);
      dims_output[1][0] = 1;
      dims_output[1][1] = state_size;
    }

    SetInputTensors(dims_input);
    SetOutputTensors(dims_output);

    /* reference version */
    auto refOut0 = make_shared_blob<ie_fp16>({Precision::FP16, dims_output[0], Layout::NC});
    refOut0->allocate();
    auto refOut1 = make_shared_blob<ie_fp16>({Precision::FP16, dims_output[1], Layout::NC});
    refOut1->allocate();
    auto refOut2 = make_shared_blob<ie_fp16>({Precision::FP16, dims_output[0], Layout::NC});
    if (param.output_num == 3) {
      refOut2 = make_shared_blob<ie_fp16>({Precision::FP16, dims_output[2], Layout::NC});
    }
    refOut2->allocate();
    // num_weights + num_bias
    auto gatesBlob = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size}, Layout::NC});
    gatesBlob->allocate();
    auto weightsBlob0_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {ngates * state_size * input_size, 1}, Layout::NC});
    weightsBlob0_repacked->allocate();
    auto weightsBlob1_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {ngates * state_size * state_size, 1}, Layout::NC});
    weightsBlob1_repacked->allocate();

    ie_fp16* h_dst = static_cast<ie_fp16*>(refOut0->buffer());
    ie_fp16* c_dst = static_cast<ie_fp16*>(refOut1->buffer());
    ie_fp16* l_h_dst = (param.output_num == 3) ? static_cast<ie_fp16*>(refOut2->buffer()) : nullptr;
    ie_fp16* gates = static_cast<ie_fp16*>(gatesBlob->buffer());

    /* weights repacking */
    ie_fp16* weights0_repacked = static_cast<ie_fp16*>(weightsBlob0_repacked->buffer());
    ie_fp16* weights1_repacked = static_cast<ie_fp16*>(weightsBlob1_repacked->buffer());

    std::map<std::string, std::string> params {{"hidden_size", std::to_string(state_size)}};

    /* weights generating */
    TBlob<uint8_t>::Ptr weightsBlob_for_net(GenWeights((num_weights + num_bias)));
    ie_fp16 *weights_for_net = static_cast<ie_fp16*>(weightsBlob_for_net->buffer());

    TBlob<uint8_t>::Ptr weightsBlob_tmp(GenWeights(num_weights + num_bias));
    ie_fp16 *weights0 = static_cast<ie_fp16*>(weightsBlob_tmp->buffer());
    ie_fp16 *weights1 = weights0 + ngates * state_size * input_size;

    TBlob<uint8_t>::Ptr weightsBlob_inv_tmp(GenWeights(num_weights + num_bias));
    ie_fp16 *weights_inv0 = static_cast<ie_fp16*>(weightsBlob_inv_tmp->buffer());
    ie_fp16 *weights_inv1 = weights_inv0 + ngates * state_size * input_size;
    ie_fp16 *bias = weights0 + num_weights;
    ie_fp16 *bias_inv = weights_inv0 + num_weights;

    int counter = 0;
    for (int j = 0; j < ngates * state_size; j++) {
        for (int i = 0; i < input_size; i++) {
            weights0[(input_size) * j + i] = PrecisionUtils::f32tof16(((float)(rand() % input_size)) / input_size * 0.01);
            weights_for_net[counter++] = weights0[(input_size) * j + i];
        }
        for (int i = 0; i < state_size; i++) {
            weights1[(state_size) * j + i] = PrecisionUtils::f32tof16(((float)(rand() % state_size)) / state_size * 0.05f);
            weights_for_net[counter++] = weights1[(state_size) * j + i];
        }
    }

    for (int i = 0; i < num_bias; i++) {
        bias[i] = PrecisionUtils::f32tof16((float)((rand() % num_bias)) / num_bias);
        *(weights_for_net + num_weights + i) = bias[i];
    }
  
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("LSTMCell")
                                        .params(params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        {},
                                        weightsBlob_for_net));
  
    auto pInputBlob = _inputMap.begin();
    Blob::Ptr inputBlob0 = pInputBlob->second;
    ie_fp16 *src_data0 = static_cast<ie_fp16*>(inputBlob0->buffer());
    pInputBlob++;
    Blob::Ptr inputBlob1 = pInputBlob->second;
    ie_fp16 *src_data1 = static_cast<ie_fp16*>(inputBlob1->buffer());
    pInputBlob++;
    Blob::Ptr inputBlob2 = pInputBlob->second;
    ie_fp16 *src_data2 = static_cast<ie_fp16*>(inputBlob2->buffer());

    // gates repacking
    {
        for (int g = 0; g < ngates; g++) {
            int stride = state_size * input_size;
            for (int i = 0; i < stride; i++) {
                weights_inv0[g * stride + i] = weights0[gate_map[g] * stride + i];
            }
        }
        for (int g = 0; g < ngates; g++) {
            int stride = state_size * state_size;
            for (int i = 0; i < stride; i++) {
                weights_inv1[g * stride + i] = weights1[gate_map[g] * stride + i];
            }
        }
        for (int g = 0; g < ngates; g++) {
            int stride = state_size;
            for (int i = 0; i < stride; i++) {
                bias_inv[g * stride + i] = bias[gate_map[g] * stride + i];
            }
        }
    }

    matrix_copy_transpose(weights_inv0, weights0_repacked, ngates * state_size, input_size);
    matrix_copy_transpose(weights_inv1, weights1_repacked, ngates * state_size, state_size);

    for (int i = 0; i < input_size; i++) {
        src_data0[i] = PrecisionUtils::f32tof16(( ((float)(rand() % input_size)) / input_size * .1f));
    }

    for (int i = 0; i < state_size; i++) {
        src_data1[i] = PrecisionUtils::f32tof16(( ((float)(rand() % state_size)) / state_size * .2f));
    }

    for (int i = 0; i < state_size; i++) {
        src_data2[i] = PrecisionUtils::f32tof16(( ((float)(rand() % state_size)) / state_size * .3f));
    }

    lstm_cell(input_size,
              state_size,

              // weights
              weights0_repacked,
              weights1_repacked,
              bias_inv,

              // input
              src_data0,
              src_data1,
              src_data2,

              param.output_num,
              // output
              h_dst,
              c_dst,
              l_h_dst,

              gates
            );
    ASSERT_TRUE(Infer());

    /* output tensors comparing */
    auto pOutputBlob = _outputMap.begin();
    auto outputBlob0 = pOutputBlob->second;
    CompareCommonAbsolute(outputBlob0, refOut0, ERROR_BOUND);
    if (param.output_num > 1) {
      pOutputBlob++;
      auto outputBlob1 = pOutputBlob->second;
      CompareCommonAbsolute(outputBlob1, refOut1, ERROR_BOUND);
    }
    if (param.output_num > 2) {
      auto outputBlob2 = pOutputBlob->second;
      CompareCommonAbsolute(outputBlob2, refOut1, ERROR_BOUND);
    }
}
