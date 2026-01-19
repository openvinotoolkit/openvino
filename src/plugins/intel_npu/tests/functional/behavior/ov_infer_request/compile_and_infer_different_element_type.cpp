// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compile_and_infer_different_element_type.hpp"

#include "common/utils.hpp"

namespace {

const std::vector<ov::AnyMap> configs = {};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestElementTypeTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestElementTypeTests>);

}  // namespace
