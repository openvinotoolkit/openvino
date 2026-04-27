// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "caching_tests.hpp"

#include <vector>

#include "common/utils.hpp"

using namespace ov::test::behavior;

namespace {

std::vector<ov::AnyMap> config = {
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, ov::util::codec_xor})},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
     ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, ov::util::codec_xor})}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompileModelLoadFromFileTestBaseNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(config)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileModelLoadFromFileTestBaseNPU>);

}  // namespace
