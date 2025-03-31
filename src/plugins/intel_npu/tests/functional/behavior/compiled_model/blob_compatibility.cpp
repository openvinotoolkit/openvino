// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <blob_compatibility.hpp>

const std::vector<ov::test::behavior::BlobPair> config = {
    // MTL
    // ov release 2025.0.0
    {"MTL", "blob_compatibility_dummy_model_MTL_ov_release_2025_0_0.blob"},
    {"MTL", "blob_compatibility_dummy_model_stateful_MTL_ov_release_2025_0_0.blob"},
    {"MTL", "blob_compatibility_dummy_model_dynamic_shapes_MTL_ov_release_2025_0_0.blob"}

    // LNL
    // ov release 2025.0.0
    {"LNL", "blob_compatibility_dummy_model_LNL_ov_release_2025_0_0.blob"},
    {"LNL", "blob_compatibility_dummy_model_stateful_LNL_ov_release_2025_0_0.blob"},
    {"LNL", "blob_compatibility_dummy_model_dynamic_shapes_LNL_ov_release_2025_0_0.blob"}};

INSTANTIATE_TEST_SUITE(smoke_Behavior_NPU,
                       OVBlobCompatibilityNPU,
                       ::testing::ValuesIn(config),
                       ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>)
