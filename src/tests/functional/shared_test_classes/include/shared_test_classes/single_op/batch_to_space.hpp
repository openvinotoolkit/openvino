// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using batchToSpaceParamsTuple = typename std::tuple<
        std::vector<int64_t>,              // block shape
        std::vector<int64_t>,              // crops begin
        std::vector<int64_t>,              // crops end
        std::vector<InputShape>,           // Input shapes
        ov::element::Type,                 // Model type
        std::string>;                      // Device name>;

class BatchToSpaceLayerTest : public testing::WithParamInterface<batchToSpaceParamsTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<batchToSpaceParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
