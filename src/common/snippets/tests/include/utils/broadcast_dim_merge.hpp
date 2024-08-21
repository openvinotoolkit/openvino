// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>


namespace ov {
namespace test {
namespace snippets {

// D1, D2, Result
using BroadcastMergeDimParams = std::tuple<size_t, size_t, size_t>;

class BroadcastMergeDimTest : public testing::TestWithParam<BroadcastMergeDimParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastMergeDimParams> obj);

protected:
    void SetUp() override;
    BroadcastMergeDimParams m_dims = {};
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
