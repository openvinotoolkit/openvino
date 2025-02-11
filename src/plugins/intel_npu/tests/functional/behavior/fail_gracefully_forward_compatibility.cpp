// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/fail_gracefully_forward_compatibility.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

bool UnsupportedTestOperation::visit_attributes(AttributeVisitor& /*visitor*/) {
    return true;
}

namespace {

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         FailGracefullyTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         FailGracefullyTest::getTestCaseName);

}  // namespace
