// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,     // Input shapes
        ov::element::Type,           // Model type
        std::vector<int64_t>,        // Shift
        std::vector<int64_t>,        // Axes
        ov::test::TargetDevice       // Device name
> rollParams;

class RollLayerTest : public testing::WithParamInterface<rollParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<rollParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
