// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using GatherTreeParamsTuple = typename std::tuple<
        ov::Shape,                         // Input tensors shape
        ov::test::utils::InputLayerType,   // Secondary input type
        ov::element::Type,                 // Model type
        std::string>;                      // Device name

class GatherTreeLayerTest : public testing::WithParamInterface<GatherTreeParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj);

protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov
