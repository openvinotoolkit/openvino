// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/model_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

// when HSM becomes the default and the legacy format and export_hsm_format are dropped, remove the
// export_hsm_format(true) configs - the default ones then cover HSM.
INSTANTIATE_TEST_SUITE_P(smoke_,
                         WeightlessCacheAccuracy,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(model_dtypes),
                                            ::testing::Values(ov::AnyMap{ov::enable_weightless("ON"),
                                                                         ov::cache_mode("OPTIMIZE_SPEED")},
                                                              ov::AnyMap{ov::cache_mode("OPTIMIZE_SIZE")},
                                                              ov::AnyMap{ov::enable_weightless("OFF")},
                                                              ov::AnyMap{ov::enable_weightless("ON"),
                                                                         ov::cache_mode("OPTIMIZE_SPEED"),
                                                                         ov::intel_cpu::export_hsm_format(true)},
                                                              ov::AnyMap{ov::cache_mode("OPTIMIZE_SIZE"),
                                                                         ov::intel_cpu::export_hsm_format(true)}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         WeightlessCacheAccuracy::get_test_case_name);

// same as above for the export_hsm_format(true) config.
INSTANTIATE_TEST_SUITE_P(smoke_,
                         WeightlessCacheAccuracyLowPrecision,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(low_precision_dtypes),
                                            ::testing::Values(ov::AnyMap{ov::enable_weightless("ON"),
                                                                         ov::cache_mode("OPTIMIZE_SPEED")},
                                                              ov::AnyMap{ov::cache_mode("OPTIMIZE_SIZE")},
                                                              ov::AnyMap{ov::cache_mode("OPTIMIZE_SIZE"),
                                                                         ov::intel_cpu::export_hsm_format(true)}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         WeightlessCacheAccuracy::get_test_case_name);
