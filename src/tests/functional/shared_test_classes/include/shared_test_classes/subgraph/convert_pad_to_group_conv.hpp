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

typedef std::tuple<ov::Shape,             // input shape
                   std::vector<int64_t>,  // pad_begin
                   std::vector<int64_t>,  // pad_end
                   float,                 // pad_value
                   ov::op::PadMode,       // pad_mode
                   std::string            // Device name
                   >
    PadParams;

class ConvertPadToConvTests : public testing::WithParamInterface<PadParams>,
                              virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PadParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
