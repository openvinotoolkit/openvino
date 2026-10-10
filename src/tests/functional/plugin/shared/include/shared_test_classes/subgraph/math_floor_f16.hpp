// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Regression test for #184635: f16 inference must keep the f16 rounding of a Math op that feeds a Floor.
//
//   Parameter(f16) -> [Multiply(f16 const)] -> MathOp -> Floor -> Result
//
// Each case uses an input for which Floor gives a different result depending on whether the value reaching
// or leaving MathOp is rounded to f16, e.g. cos(6.27734375) == 0.99998 in f32 but rounds to 1.0 in f16.
struct MathFloorF16Case {
    ov::test::utils::ActivationTypes math_type;
    float input;
    // Non-zero: Multiply the input by this constant first; used by ops whose f16 effect is on the input side.
    float pre_multiplier;
};

using MathFloorF16Params = std::tuple<MathFloorF16Case,
                                      ov::element::Type,  // inference precision hint
                                      std::string>;       // device

class MathFloorF16Test : public testing::WithParamInterface<MathFloorF16Params>,
                         virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MathFloorF16Params>& obj);
    static const std::vector<MathFloorF16Case>& all_cases();

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void compile_model() override;

    MathFloorF16Case test_case{};
};

}  // namespace test
}  // namespace ov
