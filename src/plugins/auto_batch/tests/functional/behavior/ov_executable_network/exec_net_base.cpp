// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"

namespace ov {
namespace test {
namespace behavior {
auto autoBatchConfigs =
    std::vector<ov::AnyMap>{// explicit batch size 4 to avoid fallback to no auto-batching
                            {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
                             // no timeout to avoid increasing the test time
                             {ov::auto_batch_timeout.name(), "0"}}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(autoBatchConfigs)),
                         OVCompiledModelBaseTest::getTestCaseName);

std::vector<ov::element::Type> convert_types = {ov::element::f16, ov::element::i64};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatchBehaviorTests,
                         CompiledModelSetType,
                         ::testing::Combine(::testing::ValuesIn(convert_types),
                                            ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(autoBatchConfigs)),
                         CompiledModelSetType::getTestCaseName);
}  // namespace behavior
}  // namespace test
}  // namespace ov
