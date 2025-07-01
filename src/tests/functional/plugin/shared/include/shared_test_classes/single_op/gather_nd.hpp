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

typedef std::tuple<
        std::vector<InputShape>,  // Input shapes
        ov::Shape,                // Indices shape
        int,                      // Batch dim
        ov::element::Type,        // Model type
        ov::element::Type,        // Indices type
        std::string               // Device name
> GatherNDParams;

class GatherNDLayerTest : public testing::WithParamInterface<GatherNDParams>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams> &obj);

protected:
    void SetUp() override;
};

class GatherND8LayerTest : public testing::WithParamInterface<GatherNDParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
