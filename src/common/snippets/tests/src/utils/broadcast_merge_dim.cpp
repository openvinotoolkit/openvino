// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/broadcast_dim_merge.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string BroadcastMergeDimTest::getTestCaseName(testing::TestParamInfo<BroadcastMergeDimParams> obj) {
    BroadcastMergeDimParams params = obj.param;
    std::ostringstream result;
    result << "D0=" << ov::snippets::utils::value2str(std::get<0>(params)) << "_";
    result << "D1=" << ov::snippets::utils::value2str(std::get<1>(params)) << "_";
    result << "DST=" << ov::snippets::utils::value2str(std::get<2>(params));
    return result.str();
}

void BroadcastMergeDimTest::SetUp() {
    m_dims = this->GetParam();
}

TEST_P(BroadcastMergeDimTest, BrodcastMergeDim) {
    size_t d1, d2, dst, result;
    std::tie(d1, d2, dst) = this->m_dims;
    ASSERT_TRUE(ov::snippets::utils::broadcast_merge_dim(result, d1, d2));
    ASSERT_EQ(result, dst);
}

namespace BrodcastMergeDimInstantiation {

constexpr size_t dynamic = ov::snippets::utils::get_dynamic_value<size_t>();

const std::vector<BroadcastMergeDimParams> dimension_cases = {
    {10, 10, 10},
    {10, 1, 10},
    {1, 10, 10},
    {dynamic, 10, 10},
    {10, dynamic, 10},
    {dynamic, dynamic, dynamic},
    {dynamic, 1, dynamic},
    {1, dynamic, dynamic},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BrodcastMergeDim, BroadcastMergeDimTest,
                         ::testing::ValuesIn(dimension_cases),
                         BroadcastMergeDimTest::getTestCaseName);

}  // namespace BrodcastMergeDimInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov