// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using fakeConvertParamsTuple = typename std::tuple<ov::element::Type,        // destination_type
                                                   std::vector<InputShape>,  // Input shapes
                                                   ov::element::Type,        // Model type
                                                   std::string>;             // Device name

class FakeConvertLayerTest : public testing::WithParamInterface<fakeConvertParamsTuple>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fakeConvertParamsTuple>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
