// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Operations whose result jumps at some value, so they are sensitive to f16 rounding of their input.
enum class DiscontinuousOp {
    Floor,
    Ceiling,
    RoundHalfToEven,
    RoundHalfAwayFromZero,
    Sign,
    Mod,       // Mod(math, 1)
    FloorMod,  // FloorMod(math, 1)
};

// Regression test for #184635: f16 inference must keep the f16 rounding of a Math op that feeds a discontinuous op.
//
//   Parameter(f16) -> [Multiply(f16 const)] -> MathOp -> DiscontinuousOp -> Result
//
// Each case uses an input for which the discontinuous op gives a different result depending on whether the value
// reaching or leaving MathOp is rounded to f16, e.g. cos(6.27734375) == 0.99998 in f32 but rounds to 1.0 in f16,
// so Floor gives 1 instead of 0.
struct MathDiscontinuousOpF16Case {
    ov::test::utils::ActivationTypes math_type;
    DiscontinuousOp op;
    float input;
    // Non-zero: Multiply the input by this constant first; used by ops whose f16 effect is on the input side.
    float pre_multiplier;
};

using MathDiscontinuousOpF16Params = std::tuple<MathDiscontinuousOpF16Case,
                                                ov::element::Type,  // inference precision hint
                                                std::string>;       // device

class MathDiscontinuousOpF16Test : public testing::WithParamInterface<MathDiscontinuousOpF16Params>,
                                   virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MathDiscontinuousOpF16Params>& obj);
    static const std::vector<MathDiscontinuousOpF16Case>& all_cases();

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void compile_model() override;

    MathDiscontinuousOpF16Case test_case{};
};

}  // namespace ov::test
