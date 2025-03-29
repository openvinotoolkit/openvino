// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"

using namespace ov::test::behavior;
namespace {
auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVCompiledModelBaseTest::getTestCaseName);

std::vector<ov::element::Type> convert_types = {ov::element::f16,
                                                ov::element::i64};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CompiledModelSetType,
                        ::testing::Combine(
                                ::testing::ValuesIn(convert_types),
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        CompiledModelSetType::getTestCaseName);
}  // namespace
