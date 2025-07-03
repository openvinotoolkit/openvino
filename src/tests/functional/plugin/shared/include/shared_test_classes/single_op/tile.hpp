// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::vector<int64_t> TileSpecificParams;
typedef std::tuple<
        TileSpecificParams,
        ov::element::Type,             // Model type
        std::vector<InputShape>,       // Input shapes
        ov::test::TargetDevice         // Device name
> TileLayerTestParamsSet;

class TileLayerTest : public testing::WithParamInterface<TileLayerTestParamsSet>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TileLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
