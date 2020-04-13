// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReduceMeanToAvgPool) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,227,227"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMean" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="ReduceMean" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reduce" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="227,227" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 2;
        reduce_axes[1] = 3;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToAvgPoolKeepDimsFalse) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,227,64"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMean" version="opset1">
        <data keep_dims="0" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reduce/pool" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="227,1" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reshape_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24"/>
            </blobs>
        </layer>
        <layer id="3" name="reduce" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 2;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToAvgPoolNonSpatial) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,24,12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMean" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reshape_begin_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="reduce/reshape_begin" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>288</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reduce/pool" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="3,1" output="1" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>288</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>288</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="reshape_end_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="5" name="reduce" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>288</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 1;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToAvgPoolNonSpatialHard) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,24,12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMean" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>12</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reshape_begin_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="reduce/reshape_begin" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>72</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reduce" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="72,1" output="1" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>72</dim>
                    <dim>12</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 1;
        reduce_axes[1] = 2;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToMaxPool) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,227,227"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMax" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="ReduceMean" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reduce" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="227,227" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="max" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 2;
        reduce_axes[1] = 3;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToMaxPoolKeepDimsFalse) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,227,64"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMax" version="opset1">
        <data keep_dims="0" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reduce/pool" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="227,1" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="max" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reshape_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="24"/>
            </blobs>
        </layer>
        <layer id="3" name="reduce" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 2;
    });
}

TEST_F(NGraphReaderTests, ReduceMeanToMaxPoolNonSpatial) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,24,12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceMax" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reshape_begin_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="reduce/reshape_begin" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>288</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reduce/pool" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="3,1" output="1" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="max" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>288</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>288</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="reshape_end_shape" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="5" name="reduce" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>288</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 1;
    });
}

TEST_F(NGraphReaderTests, ReduceSumToAvgPool) {
    std::string model = R"V0G0N(
<net name="ReduceMean" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3,227,227"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const" version="opset1">
            <data offset="0" size="16"/>
            <output>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" type="ReduceSum" version="opset1">
        <data keep_dims="1" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer  id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="ReduceMean" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="reduce/pool" precision="FP32" type="Pooling">
            <data dilations="1,1" group="1" kernel="227,227" output="3" pads_begin="0,0" pads_end="0,0" strides="1,1" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="reduce" precision="FP32" type="Power">
            <data power="1.000000" scale="51529" shift="0.000000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 139840, [](Blob::Ptr& weights) {
        auto* reduce_axes = weights->buffer().as<int64_t*>();
        reduce_axes[0] = 2;
        reduce_axes[1] = 3;
    });
}
