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
using ConvertColorI420ParamsTuple = std::tuple<
        std::vector<InputShape>,                       // Input Shape
        ov::element::Type,                             // Element type
        bool,                                          // Conversion type
        bool,                                          // 1 or 3 planes
        std::string>;                                  // Device name

class ConvertColorI420LayerTest : public testing::WithParamInterface<ConvertColorI420ParamsTuple>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertColorI420ParamsTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov
