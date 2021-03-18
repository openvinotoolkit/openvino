// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include "common_test_utils/data_utils.hpp"

TEST_F(NGraphReaderTests, LSTMSeqNetwork) {
    std::string model = R"V0G0N(
    <net name="LSTMSeqNetwork" version="10">
        <layers>
            <layer id="0" name="0" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,3,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="1" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="2" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="3" type="Parameter" version="opset1">
                <data element_type="f32" shape="10"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="4" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,1024,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="5" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,1024,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="6" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,1024"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>1024</dim>
                    </port>
                </output>
            </layer>
            <layer id="7" name="layer/LSTMSequence" type="LSTMSequence" version="opset5">
                <data hidden_size="256"/>
                <input>
                    <port id="0">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="3">
                        <dim>10</dim>
                    </port>
                    <port id="4">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>256</dim>
                    </port>
                    <port id="6">
                        <dim>1</dim>
                        <dim>1024</dim>
                    </port>
                </input>
                <output>
                    <port id="7" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                    <port id="8" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="9" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="8" name="8" type="Result" version="opset1">
                <input>
                    <port id="0"  precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
            <layer id="9" name="9" type="Result" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
            <layer id="10" name="10" type="Result" version="opset1">
                <input>
                    <port id="0"  precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="7" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="7" to-port="2"/>
            <edge from-layer="3" from-port="0" to-layer="7" to-port="3"/>
            <edge from-layer="4" from-port="0" to-layer="7" to-port="4"/>
            <edge from-layer="5" from-port="0" to-layer="7" to-port="5"/>
            <edge from-layer="6" from-port="0" to-layer="7" to-port="6"/>
            <edge from-layer="7" from-port="7" to-layer="8" to-port="0"/>
            <edge from-layer="7" from-port="8" to-layer="9" to-port="0"/>
            <edge from-layer="7" from-port="9" to-layer="10" to-port="0"/>
        </edges>
    </net>
)V0G0N";

    Blob::CPtr blob;
    Core reader;
    reader.ReadNetwork(model, blob);
}

TEST_F(NGraphReaderTests, GRUSeqNetwork) {
    std::string model = R"V0G0N(
    <net name="GRUSeqNetwork" version="10">
        <layers>
            <layer id="0" name="0" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,3,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="1" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="3" type="Parameter" version="opset1">
                <data element_type="f32" shape="10"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="4" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,768,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>768</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="5" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,768,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>768</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="6" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,768"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>768</dim>
                    </port>
                </output>
            </layer>
            <layer id="7" name="layer/LSTMSequence" type="GRUSequence" version="opset5">
                <data hidden_size="256"/>
                <input>
                    <port id="0">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="3">
                        <dim>10</dim>
                    </port>
                    <port id="4">
                        <dim>1</dim>
                        <dim>768</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5">
                        <dim>1</dim>
                        <dim>768</dim>
                        <dim>256</dim>
                    </port>
                    <port id="6">
                        <dim>1</dim>
                        <dim>768</dim>
                    </port>
                </input>
                <output>
                    <port id="7" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                    <port id="8" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="8" name="8" type="Result" version="opset1">
                <input>
                    <port id="0"  precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
            <layer id="9" name="9" type="Result" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="7" to-port="1"/>
            <edge from-layer="3" from-port="0" to-layer="7" to-port="3"/>
            <edge from-layer="4" from-port="0" to-layer="7" to-port="4"/>
            <edge from-layer="5" from-port="0" to-layer="7" to-port="5"/>
            <edge from-layer="6" from-port="0" to-layer="7" to-port="6"/>
            <edge from-layer="7" from-port="7" to-layer="8" to-port="0"/>
            <edge from-layer="7" from-port="8" to-layer="9" to-port="0"/>
        </edges>
    </net>
)V0G0N";

    Blob::CPtr blob;
    Core reader;
    reader.ReadNetwork(model, blob);
}

TEST_F(NGraphReaderTests, RNNSeqNetwork) {
    std::string model = R"V0G0N(
    <net name="RNNSeqNetwork" version="10">
        <layers>
            <layer id="0" name="0" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,3,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="1" type="Parameter" version="opset1">
                <data element_type="f32" shape="10,1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="3" type="Parameter" version="opset1">
                <data element_type="f32" shape="10"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="4" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="5" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="6" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="7" name="layer/LSTMSequence" type="RNNSequence" version="opset5">
                <data hidden_size="256"/>
                <input>
                    <port id="0">
                        <dim>10</dim>
                        <dim>3</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="3">
                        <dim>10</dim>
                    </port>
                    <port id="4">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>256</dim>
                    </port>
                    <port id="6">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="7" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                    <port id="8" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="8" name="8" type="Result" version="opset1">
                <input>
                    <port id="0"  precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
            <layer id="9" name="9" type="Result" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>10</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="7" to-port="1"/>
            <edge from-layer="3" from-port="0" to-layer="7" to-port="3"/>
            <edge from-layer="4" from-port="0" to-layer="7" to-port="4"/>
            <edge from-layer="5" from-port="0" to-layer="7" to-port="5"/>
            <edge from-layer="6" from-port="0" to-layer="7" to-port="6"/>
            <edge from-layer="7" from-port="7" to-layer="8" to-port="0"/>
            <edge from-layer="7" from-port="8" to-layer="9" to-port="0"/>
        </edges>
    </net>
)V0G0N";

    Blob::CPtr blob;
    Core reader;
    reader.ReadNetwork(model, blob);
}