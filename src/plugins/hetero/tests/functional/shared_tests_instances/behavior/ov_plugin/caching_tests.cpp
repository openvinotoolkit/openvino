// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"

using namespace ov;
using namespace ov::test::behavior;

namespace {
    static const std::vector<ov::element::Type> precisionsCPU = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
    };

    static const std::vector<ov::element::Type> floatPrecisionsCPU = {
            ov::element::f32,
            ov::element::f16,
    };

    static const std::vector<std::size_t> batchSizesCPU = {
            1, 2
    };

    const std::vector<ov::AnyMap> autoConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_CachingSupportCase, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getNumericAnyTypeFunctions()),
                                    ::testing::ValuesIn(precisionsCPU),
                                    ::testing::ValuesIn(batchSizesCPU),
                                    ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                    ::testing::ValuesIn(autoConfigs)),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_CachingSupportCase_Float, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getFloatingPointOnlyFunctions()),
                                    ::testing::ValuesIn(floatPrecisionsCPU),
                                    ::testing::ValuesIn(batchSizesCPU),
                                    ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                    ::testing::ValuesIn(autoConfigs)),
                            CompileModelCacheTestBase::getTestCaseName);
} // namespace
