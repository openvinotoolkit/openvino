// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/add_output.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/plugin_cache.hpp"

InferenceEngine::CNNNetwork getTargetNetwork() {
    auto model = FuncTestUtils::TestModel::getModelWithMemory(InferenceEngine::Precision::FP32);
    auto ie = PluginCache::get().ie();
    return ie->ReadNetwork(model.model_xml_str, model.weights_blob);
}
addOutputsParams testCases[] = {addOutputsParams(getTargetNetwork(), {"Memory_1"}, CommonTestUtils::DEVICE_GNA)};

INSTANTIATE_TEST_CASE_P(AddOutputBasic, AddOutputsTest, ::testing::ValuesIn(testCases), AddOutputsTest::getTestCaseName);
