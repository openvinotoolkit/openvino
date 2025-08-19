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
using ResultTestParamSet = std::tuple<
    std::vector<size_t>,           // Input shapes
    ov::element::Type,             // Model type
    ov::test::TargetDevice         // Device name
>;

class ResultLayerTest : public testing::WithParamInterface<ResultTestParamSet>,
                         virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ResultTestParamSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
