// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/memory_states.hpp"
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
        <layer id="1" name="Input_1" precision="FP32" type="input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Eltwise_1" precision="FP32" type="Eltwise">
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
        <layer id="3" name="Memory_2" precision="FP32" type="Memory">
            <data id="c_1-3" index="1" size="2" />
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Eltwise_2" precision="FP32" type="Eltwise">
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
        <layer id="5" name="Memory_3" precision="FP32" type="Memory">
            <data id="c_1-3" index="0" size="2" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
        </layer>
        <layer id="6" name="Activation_1" precision="FP32" type="Activation">
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
        <layer id="7" name="Memory_4" precision="FP32" type="Memory">
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
        <edge from-layer="2" from-port="2" to-layer="4" to-port="1" />
        <edge from-layer="3" from-port="0" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
        <edge from-layer="4" from-port="2" to-layer="6" to-port="0" />
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0" />
    </edges>
</net>
)V0G0N";

InferenceEngine::CNNNetwork getNetwork() {
    auto ie = PluginCache::get().ie();
    return ie->ReadNetwork(model, InferenceEngine::Blob::Ptr{});
}
std::vector<memoryStateParams> memoryStateTestCases = {
        memoryStateParams(getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_GNA)
};

INSTANTIATE_TEST_CASE_P(smoke_VariableStateBasic, VariableStateTest,
        ::testing::ValuesIn(memoryStateTestCases),
        VariableStateTest::getTestCaseName);
