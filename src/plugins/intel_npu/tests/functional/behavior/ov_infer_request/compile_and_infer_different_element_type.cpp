// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compile_and_infer_different_element_type.hpp"

#include <vector>

#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

const std::vector<ov::AnyMap> configs = {
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN), {"NPU_COMPILATION_MODE", "DefaultHW"}},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER), {"NPU_COMPILATION_MODE", "DefaultHW"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestElementTypeTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestElementTypeTests>);

}  // namespace
