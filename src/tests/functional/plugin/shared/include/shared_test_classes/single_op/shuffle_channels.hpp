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
typedef std::tuple<
        int, // axis
        int  // group
> shuffleChannelsSpecificParams;

typedef std::tuple<
        shuffleChannelsSpecificParams,
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shapes
        ov::test::TargetDevice          // Device name
> shuffleChannelsLayerTestParamsSet;

class ShuffleChannelsLayerTest : public testing::WithParamInterface<shuffleChannelsLayerTestParamsSet>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<shuffleChannelsLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
