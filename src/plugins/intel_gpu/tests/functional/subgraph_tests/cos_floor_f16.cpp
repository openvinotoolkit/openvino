// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Test for #184635: Incorrect Floor result (0.0 instead of 1.0)
// in float16 inference when Cos output approaches 1.0.
// This guards the GPU ConvertPrecision pipeline that preserves f16 rounding
// at Math op boundaries during f16 inference.

#include "shared_test_classes/subgraph/cos_floor_f16.hpp"

namespace ov::test {

class CosFloorF16GPUTest : public CosFloorF16TestBase {
public:
    void SetUp() override {
        CosFloorF16TestBase::SetUp();
        targetDevice = ov::test::utils::DEVICE_GPU;
    }
};

TEST_F(CosFloorF16GPUTest, CompareWithRefs) {
    check_floor_result();
}

}  // namespace ov::test
