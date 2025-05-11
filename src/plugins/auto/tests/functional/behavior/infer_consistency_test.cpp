// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_consistency_test.hpp"

using namespace ov::auto_plugin::tests;
namespace {
auto props = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities("MOCK_GPU", "MOCK_CPU")},
                                   {ov::device::priorities("MOCK_GPU")},
                                   {ov::device::priorities("MOCK_CPU", "MOCK_GPU")}};
};

const std::vector<bool> get_vs_set{true, false};

const std::vector<std::string> target_device{"AUTO", "MULTI"};

INSTANTIATE_TEST_SUITE_P(AutoFuncTests,
                         Consistency_Test,
                         ::testing::Combine(::testing::ValuesIn(target_device),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(props())),
                         Consistency_Test::getTestCaseName);
}  // namespace
