// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using EyeLayerTestParams = std::tuple<
    std::vector<ov::Shape>,  // eye shape
    std::vector<int>,        // output batch shape
    std::vector<int>,        // eye params (rows, cols, diag_shift)
    ov::element::Type,     // Model type
    std::string>;            // Device name

class EyeLayerTest : public testing::WithParamInterface<EyeLayerTestParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLayerTestParams> obj);
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
