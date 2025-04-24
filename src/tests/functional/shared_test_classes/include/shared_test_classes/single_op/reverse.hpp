// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using reverseParams = std::tuple<
    std::vector<size_t>,             // Input shape
    std::vector<int>,                // Axes
    std::string,                     // Mode
    ov::element::Type,               // Model type
    ov::test::TargetDevice           // Device name
>;

class ReverseLayerTest : public testing::WithParamInterface<reverseParams>,
                         virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reverseParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
