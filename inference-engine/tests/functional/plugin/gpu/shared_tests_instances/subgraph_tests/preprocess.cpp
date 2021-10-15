// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/subgraph/preprocess.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

using namespace ov::builder::preprocess;

inline std::vector<preprocess_func> GPU_smoke_preprocess_functions() {
    return std::vector<preprocess_func>{
            preprocess_func(mean_only, "mean_only", 0.01f),
            preprocess_func(scale_only, "scale_only", 0.01f),
            preprocess_func(convert_element_type_and_mean, "convert_element_type_and_mean", 0.01f),
            preprocess_func(two_inputs_basic, "two_inputs_basic", 0.01f),
            preprocess_func(two_inputs_trivial, "two_inputs_trivial", 0.01f),
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess_GPU, PrePostProcessTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(GPU_smoke_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         PrePostProcessTest::getTestCaseName);

}  // namespace
