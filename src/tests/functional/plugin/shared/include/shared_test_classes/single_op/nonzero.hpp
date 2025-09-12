// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using NonZeroLayerTestParamsSet = typename std::tuple<
    std::vector<InputShape>,             // Input shapes
    ov::element::Type,                   // Model shape
    std::string,                         // Device name
    std::map<std::string, std::string>>; // Additional network configuration

class NonZeroLayerTest : public testing::WithParamInterface<NonZeroLayerTestParamsSet>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NonZeroLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
