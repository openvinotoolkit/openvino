// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using spaceToBatchParamsTuple = typename std::tuple<
        std::vector<int64_t>,               // block_shape
        std::vector<int64_t>,               // pads_begin
        std::vector<int64_t>,               // pads_end
        std::vector<InputShape>,            // Input shapes
        ov::element::Type,                  // Model type
        std::string>;                       // Device name

class SpaceToBatchLayerTest : public testing::WithParamInterface<spaceToBatchParamsTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
