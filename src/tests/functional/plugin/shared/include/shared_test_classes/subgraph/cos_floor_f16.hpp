// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Regression test for #184635: Incorrect Floor result (0.0 instead of 1.0) in float16
// inference when Cos output approaches 1.0.
//
//   Parameter(f16) -> Cos -> Floor -> Result
//
// 6.27734375 is close to 2*pi: cos(6.27734375) == 0.99998 in f32, which stays < 1.0 (Floor
// would give 0.0). Rounded to f16 it becomes exactly 1.0 (Floor should give 1.0). This guards
// the plugin-specific ConvertPrecision handling that preserves f16 rounding at Math op
// boundaries feeding a Floor.
//
// The Template reference computes entirely in f32 and would report the wrong expected value
// (0.0), so this class intentionally does not use the standard run()/reference-comparison
// flow: derived TEST_F bodies must call check_floor_result() instead of run().
class CosFloorF16TestBase : public SubgraphBaseStaticTest {
public:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    // Compiles `function` on `targetDevice` (with `configuration` already populated by the
    // derived class if needed), infers the fixed input, and asserts the output is 1.0.
    void check_floor_result();

    static constexpr float input_value = 6.27734375f;
};

}  // namespace test
}  // namespace ov
