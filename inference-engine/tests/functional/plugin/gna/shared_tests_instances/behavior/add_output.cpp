// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/add_output.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/plugin_cache.hpp"

static const char model[] = R"V0G0N(
<net name="model" version="6">
    <layers>
        <layer id="0" name="Memory_1" precision="FP32" type="Memory">
            <data id="r_1-3" index="1" size="2" />
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Input_2" precision="FP32" type="input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Eltwise_3" precision="FP32" type="Eltwise">
            <data operation="mul" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Activation_4" precision="FP32" type="Activation">
            <data type="sigmoid" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Memory_5" precision="FP32" type="Memory">
            <data id="r_1-3" index="0" size="2" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
    </edges>
</net>
)V0G0N";

InferenceEngine::CNNNetwork getTargetNetwork() {
    auto ie = PluginCache::get().ie();
    return ie->ReadNetwork(model, InferenceEngine::Blob::Ptr{});
}

std::vector<addOutputsParams> testCases = {
        addOutputsParams(getTargetNetwork(), {"Memory_1"}, CommonTestUtils::DEVICE_GNA)
};

INSTANTIATE_TEST_CASE_P(smoke_AddOutputBasic, AddOutputsTest,
        ::testing::ValuesIn(testCases),
        AddOutputsTest::getTestCaseName);
