// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits.h>
#include "behavior/ov_infer_request/iteration_chaining.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
    { ov::hint::inference_precision(ov::element::f32) }
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVIterationChaining,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs)),
                        OVIterationChaining::getTestCaseName);

}  // namespace
