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

typedef std::tuple<ov::Shape,             // Input shapes
                   size_t,                // Axis
                   std::vector<size_t>,   // Split number
                   std::vector<size_t>,   // Index connected layer
                   std::vector<int64_t>,  // Pad begin
                   std::vector<int64_t>,  // Pad end
                   ov::op::PadMode,       // Pad mode
                   ov::element::Type,     // Input element type
                   std::string            // Device name
                   >
    SplitPadTuple;

class VariadicSplitPad : public testing::WithParamInterface<SplitPadTuple>,
                         virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitPadTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
