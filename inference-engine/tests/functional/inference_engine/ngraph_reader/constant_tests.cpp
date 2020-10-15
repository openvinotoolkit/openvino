// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <legacy/ie_util_internal.hpp>
#include "ngraph_reader_tests.hpp"

using namespace InferenceEngine;

TEST_F(NGraphReaderTests, ReadConstantNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="constant" type="Const" version="opset1">
            <data offset="0" size="5808"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

        IE_SUPPRESS_DEPRECATED_START
        Core ie;
        Blob::Ptr weights;

        weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {5808}, Layout::C));
        weights->allocate();

        auto network = ie.ReadNetwork(model, weights);
        auto clonedNetwork = cloneNetwork(network);
        auto clonedNet = cloneNet(network);

        IE_SUPPRESS_DEPRECATED_END
}
