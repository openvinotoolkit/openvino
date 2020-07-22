// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <inference_engine.hpp>


TEST(IRReaderTest, ThrowIfIRVersionLessThan10) {
    InferenceEngine::Core ie;

    static char const *model = R"V0G0N(<net name="Network" version="7" some_attribute="Test Attribute">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::Blob::CPtr weights;
    ASSERT_THROW(ie.ReadNetwork(model, weights), InferenceEngine::details::InferenceEngineException);
}
