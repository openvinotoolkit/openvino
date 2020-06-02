// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ConvertBroadcastToTiles1) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="112,1"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const" version="opset1">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1:" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1:_3" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 6422528, [](Blob::Ptr& weights) {
                auto * broadcast1_shape = reinterpret_cast<int64_t *>(weights->buffer().as<int8_t*>() + 256);
                broadcast1_shape[0] = 1;
                broadcast1_shape[1] = 64;
                broadcast1_shape[2] = 112;
                broadcast1_shape[3] = 112;
            });
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles2) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const" version="opset1">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1:" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1:_3" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="broadcast_1:_3_2" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 6422528, [](Blob::Ptr& weights) {
                auto * broadcast1_shape = reinterpret_cast<int64_t *>(weights->buffer().as<int8_t*>() + 256);
                broadcast1_shape[0] = 1;
                broadcast1_shape[1] = 64;
                broadcast1_shape[2] = 112;
                broadcast1_shape[3] = 112;
            });
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles3) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,64,1,112"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const" version="opset1">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 6422528, [](Blob::Ptr& weights) {
                auto * broadcast1_shape = reinterpret_cast<int64_t *>(weights->buffer().as<uint8_t *>() + 256);
                broadcast1_shape[0] = 1;
                broadcast1_shape[1] = 64;
                broadcast1_shape[2] = 112;
                broadcast1_shape[3] = 112;
            });
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles4) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="3,64"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const" version="opset1">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axes" type="Const" version="opset1">
            <data offset="288" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 6422528, [](Blob::Ptr& weights) {
        auto * broadcast1_shape = reinterpret_cast<int64_t *>(weights->buffer().as<int8_t*>() + 256);
        broadcast1_shape[0] = 1;
        broadcast1_shape[1] = 3;
        broadcast1_shape[2] = 64;
        broadcast1_shape[3] = 64;

        broadcast1_shape[4] = 1;
        broadcast1_shape[5] = 2;
    });
}

TEST_F(NGraphReaderTests, DISABLED_ConvertBroadcastToTiles5) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter" version="opset1">
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const" version="opset1">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axes" type="Const" version="opset1">
            <data offset="288" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_2" precision="FP32" type="Tile">
            <data axis="1" tiles="3"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 6422528, [](Blob::Ptr& weights) {
        auto * broadcast1_shape = reinterpret_cast<int64_t *>(weights->buffer().as<int8_t*>() + 256);
        broadcast1_shape[0] = 1;
        broadcast1_shape[1] = 3;
        broadcast1_shape[2] = 64;
        broadcast1_shape[3] = 64;

        broadcast1_shape[4] = 1;
        broadcast1_shape[5] = 2;
    });
}
