// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class BroadcastEltwiseEliminated : public testing::WithParamInterface<const char*>,
                                   public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<const char*> &obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace test
}  // namespace ov
