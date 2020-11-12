// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/memory_states.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/plugin_cache.hpp"

InferenceEngine::CNNNetwork getNetwork() {
    auto model = FuncTestUtils::TestModel::getModelWithMultipleMemoryConnections(InferenceEngine::Precision::FP32);
    auto ie = PluginCache::get().ie();
    return ie->ReadNetwork(model.model_xml_str, model.weights_blob);
}
std::vector<memoryStateParams> memoryStateTestCases = {
        memoryStateParams(getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_CPU)
};

INSTANTIATE_TEST_CASE_P(smoke_VariableStateBasic, VariableStateTest,
        ::testing::ValuesIn(memoryStateTestCases),
        VariableStateTest::getTestCaseName);
