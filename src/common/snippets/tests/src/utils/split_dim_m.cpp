// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/split_dim_m.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string SplitDimensionMTest::getTestCaseName(testing::TestParamInfo<SplitDimensionMParams> obj) {
    const auto& input = obj.param.input;
    const auto& reference = obj.param.reference;
    std::ostringstream result;
    result << "Batch=" << input.cur_batch << "_";
    result << "CurM=" << input.cur_m << "_";
    result << "OptimalParallelWorkAmount=" << input.concurrency << "_";
    result << "IsSplit=" << reference.is_split << "_";
    result << "BatchM=" << reference.batch_m << "_";
    result << "KernelM=" << reference.kernel_m;
    return result.str();
}

TEST_P(SplitDimensionMTest, SplitDimensionM) {
    const auto& input = GetParam().input;
    const auto& reference = GetParam().reference;

    // last_dim is fixed since it doesn't affect the SplitDimensionM result.
    static const size_t last_dim = 1024;
    ov::Shape shape = {input.cur_batch, input.cur_m, last_dim};
    size_t batch_m_dim, new_m_dim;
    bool result = ov::snippets::pass::SplitDimensionM::split(shape,
                                                             input.concurrency,
                                                             batch_m_dim,
                                                             new_m_dim);

    ASSERT_EQ(result, reference.is_split);
    if (result) {
        ASSERT_EQ(batch_m_dim, reference.batch_m);
        ASSERT_EQ(new_m_dim, reference.kernel_m);
    }
}

namespace SplitDimensionMInstantiation {
const std::vector<SplitDimensionMParams> split_dimension_cases = {
    // Negative test cases: split is not needed
    {InputData{40 /*cur_batch*/, 32 /*cur_m*/, 40 /*concurrency*/}, ReferenceData{false /*is_split*/}},
    {InputData{65, 32, 40}, ReferenceData{false}},

    // Positive test cases
    {InputData{20 /*cur_batch*/, 32 /*cur_m*/, 40 /*concurrency*/}, ReferenceData{true /*is_split*/, 2 /*batch_m*/, 16 /*kernel_m*/}},
    {InputData{30, 60, 40}, ReferenceData{true, 2, 30}},
    {InputData{10, 100, 40}, ReferenceData{true, 4, 25}},
    {InputData{15, 45, 40}, ReferenceData{true, 5, 9}},
    {InputData{25, 50, 40}, ReferenceData{true, 2, 25}},
    {InputData{5, 16384, 40}, ReferenceData{true, 8, 2048}},
    {InputData{5, 16384, 32}, ReferenceData{true, 32, 512}},
    {InputData{48, 4097, 32}, ReferenceData{true, 17, 241}},
    {InputData{48, 6600, 32}, ReferenceData{true, 200, 33}},
    {InputData{12, 128, 16}, ReferenceData{true, 4, 32}},
    {InputData{16, 384, 60}, ReferenceData{true, 12, 32}},
    {InputData{16, 384, 24}, ReferenceData{true, 12, 32}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SplitDimensionM,
                         SplitDimensionMTest,
                         ::testing::ValuesIn(split_dimension_cases),
                         SplitDimensionMTest::getTestCaseName);

}  // namespace SplitDimensionMInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov